# Conversation Memory Management: Model-Directed Block Operations

## Problem

At high context pressure (>70%), the dominant consumer is conversation
history — not tool results. The proxy already manages tool results via
age-based eviction. But conversation text (user messages, assistant
responses, pasted content) is untouchable. The model has no mechanism
to say "I don't need turns 50-100 verbatim anymore."

The existing phantom tools (memory_release, memory_fault) operate on
tool results only. The conversation layer — which can be 70-80% of
context at high pressure — has no management interface.

## Design

### Architecture

```
  Claude Code ←→ Pichay Proxy ←→ Anthropic API
                     ↕
              Content-Hash KV Store
```

The proxy intercepts both directions:
- **Inbound** (request to API): label conversation blocks with short IDs
- **Outbound** (response to Claude Code): scan for cleanup tags, execute
  operations, strip tags before forwarding

### Inbound: Block Labeling

On each request, the proxy hashes each message block's content and
assigns a short stable ID. These IDs are injected as lightweight
markers visible to the model:

```
[block:a3f2] User message about experimental design...
[block:7e9d] Assistant response with corpus analysis results...
```

The KV store maps: `block_id → {content_hash, original_content, size, turn}`

If the framework rewrites a block between turns, the content hash
changes. The proxy detects the mismatch and re-labels rather than
acting on a stale reference.

### Outbound: Cleanup Tags

The model includes structured tags in its response when it wants to
manage its context:

```xml
<memory_cleanup>
  <op block="a3f2" action="drop" />
  <op block="7e9d" action="anchor" />
  <op block="b1c4" action="summarize" summary="Corpus analysis showed
    75.6% of tool results are never re-referenced. Median re-reference
    distance is 111 turns. Read results dominate the working set." />
</memory_cleanup>
```

The proxy:
1. Parses the cleanup tags from the assistant response
2. Validates each block_id exists in the KV store
3. Executes the operations on the stored content
4. Strips the `<memory_cleanup>` tags before forwarding to Claude Code
5. On the next request, replaces the original blocks with their
   compressed forms

### Operations

**drop**: Remove entirely. No anchor, no recovery. For content that is
truly dead (pasted conversations, superseded discussions, resolved
tangents). The block disappears from the next request.

**anchor**: Replace with a minimal marker:
```
[Compressed block a3f2: user message about experimental design
(4,200 tokens). Use memory_fault with id 'a3f2' to restore.]
```
The model knows something was here and can fault it back. The full
content stays in the KV store.

**summarize**: Replace with a model-authored summary + fault handle:
```
[Compressed block 7e9d (was 8,100 tokens, now 350): Corpus analysis
across 63 sessions, 9,911 tool results. 75.6% never re-referenced.
Median re-reference distance 111 turns. Read results are 75% of
working set. Use memory_fault with id '7e9d' to restore full content.]
```
The summary is written by the model (who knows what matters), not by
the proxy (who doesn't). The full content stays in the KV store.

### Security: Tag Sanitization

**Attack surface**: If user content contains `<memory_cleanup>` tags,
the proxy could be tricked into dropping conversation blocks.

**Mitigation**: The proxy scans in both directions:
- **Inbound**: Strip any `<memory_cleanup>` tags from user messages
  and system prompt before forwarding. User content cannot command
  the proxy.
- **Outbound**: Only process `<memory_cleanup>` tags in assistant
  responses. Validate block_ids against the KV store (unknown IDs
  are ignored, not acted on).

Additionally, the tags are stripped from the response before Claude
Code sees them, so they don't accumulate in the conversation history.

### KV Store Design

Per-session, in-memory, keyed by content hash:

```python
@dataclass
class BlockEntry:
    block_id: str           # Short ID (4-8 hex chars)
    content_hash: str       # SHA-256 of original content
    original_content: str   # Full content for fault restoration
    size: int               # Token estimate or byte count
    turn: int               # Turn when first seen
    status: str             # "resident" | "anchored" | "summarized" | "dropped"
    summary: str | None     # Model-authored summary (if summarized)
```

The store maps both `block_id → BlockEntry` and `content_hash → block_id`
for deduplication and stale-detection.

### Integration with Existing Systems

- **memory_fault**: Extended to accept block_ids. Restores anchored or
  summarized blocks to full content from the KV store.
- **System status block**: Extended to include block inventory at high
  pressure: "12 conversation blocks, 3 are >5k tokens, oldest from
  turn 45."
- **Eviction policy**: Conversation blocks use different thresholds
  than tool results. Large blocks at high pressure get labeled with
  size hints to prompt the model to act.

### Model Prompt Updates

The Pichay system status block should include guidance:

```
When context pressure is high, you can manage conversation memory by
including <memory_cleanup> tags in your response. Reference blocks by
their [block:xxxx] labels. Operations: drop (remove permanently),
anchor (replace with fault handle), summarize (replace with your
summary + fault handle). Prioritize dropping or summarizing large
blocks from resolved topics.
```

## What This Enables

1. **Model-directed memory management**: The model decides what to
   keep, summarize, or discard — not a fixed age policy.
2. **Conversation-layer compression**: The dominant context consumer
   becomes manageable.
3. **Declared losses**: Every compression is explicit — the model
   writes the summary, the anchor records what was lost.
4. **Extended effective context**: A 200k window with conversation
   management could support sessions that would otherwise require
   500k+ of raw context.
5. **Self-extending sessions**: As pressure rises, the model
   compresses old material, creating headroom for new work.
   Sessions could theoretically run indefinitely.

## Implementation Phases

**Phase 1**: KV store + block labeling (inbound only). No cleanup
tags yet. Establishes the content-hash infrastructure and lets us
measure block sizes and lifetimes.

**Phase 2**: Cleanup tag processing (outbound). The model can drop
and anchor blocks. Summary operation deferred (requires the model
to write summaries inline, which costs output tokens).

**Phase 3**: Async summarization. Instead of the model writing
summaries inline, the proxy sends blocks to a background API call
for summarization. The model just says "summarize block a3f2" and
the proxy handles the rest asynchronously.

**Phase 4**: Proactive suggestions. The system status block includes
recommendations: "Block a3f2 (8,100 tokens, turn 45, topic: experimental
design) is a candidate for summarization." The model confirms or
overrides.

## Open Questions

- Block granularity: per-message or per-content-block? Per-message is
  simpler but coarser. Per-content-block allows finer control but more
  labels.
- Label format: `[block:xxxx]` inline in text vs. separate metadata
  block? Inline is visible to the model but costs tokens. Metadata
  is invisible but requires the model to reference blocks by position.
- Output token cost: cleanup tags in the response consume output tokens.
  Is this cost justified by the context savings? (Almost certainly yes
  for large blocks, but the model needs to learn the trade-off.)
- Interaction with framework compaction: if Claude Code's own compaction
  rewrites messages, the content hashes change. The proxy needs to
  detect and re-label gracefully.
- KV store persistence: currently per-session (in-memory). Should it
  persist across proxy restarts? Probably not for the research prototype.
