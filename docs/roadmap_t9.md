# Pichay Roadmap — Post-T9 Session (2026-03-02)

Generated from a 3+ hour production session running through the proxy.
Session context: two papers strengthened, five research agents, proxy
feature shipped (count_tokens forwarding), dashboard built, working
set curve generated. Session ran at 44-69% capacity throughout —
never hit compaction.

## What We Learned Today

- **53% byte reduction** in production (416KB → 196KB), better than
  paper's 37.1% from controlled experiment
- **Working set stabilizes** at ~300KB with transient spikes to 600KB+
  on large file reads, recovering within 2-3 turns
- **Fault rate**: 0.18% (8/4495 evictions) — higher than replay
  simulation (0.025%) because production agents re-read files
- **Silent degradation is real**: wrote a bug because evicted file
  contents led to wrong API call. Indistinguishable from baseline
  hallucination. N=1 but the category is important.
- **Conversation messages are the monotonic floor**: tool results get
  evicted, conversation history doesn't. The floor grows ~2-5KB per
  exchange. That's what eventually kills the session.
- **count_tokens endpoint**: Claude Code polls this periodically, not
  just on /context. Proxy now forwards it with compaction applied,
  so Claude Code sees post-eviction working set size.
- **CLAUDE_AUTOCOMPACT_PCT_OVERRIDE**: env var controls client-side
  compaction threshold. Combined with proxy, could defer client
  compaction almost entirely.

## Engineering Priorities

### E1: Conversation Message Eviction
Highest leverage. Currently the proxy only evicts tool results.
Conversation messages (user + assistant turns) grow monotonically.
Summarize old exchanges the same way tool results get summarized.
This is what makes the floor bounded and sessions potentially
unbounded.

### E2: Tool Classification for Fault Handling
Not all faults are equal:
- **Read**: content may have changed since eviction (file edits)
- **Bash**: result is ALWAYS ephemeral (different output each time)
- **Glob/Grep**: result depends on world state at query time
- Currently: all faults correctly forward to real tool (no stale
  cache serving). But classification could inform eviction priority —
  Bash results are safer to evict early because restoration cost is
  just a re-run.

### E3: Launch Script / Turnkey Install
Two env vars is the install:
```
CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=95 \
ANTHROPIC_BASE_URL=http://localhost:PORT \
claude
```
Package as `pipx install pichay` or similar. Default to observation-
only mode with --compact flag for intervention.

### E4: Live Dashboard
Current: stderr status line per request + post-hoc matplotlib plot.
Needed: live terminal dashboard (curses) or web view showing:
- Working set curve (real-time)
- Fault rate (running)
- Per-tool eviction stats
- Token cap utilization %
Script exists at `tools/dashboard.py` but generates static PNG.

### E5: JSONL Schema Cleanup
Current schema has inconsistent nesting:
- `messages.messages_total_bytes` under request records
- `messages_bytes_before/after` flat under compaction records
- Different field names for same concept
Flatten and version the schema.

## Research Priorities

### R1: Working Set Curve Analysis
The working set stabilizes. Characterize the steady state formally:
- Fit parametric model (logarithmic? square root?)
- Identify the characteristic working set size
- Determine if it varies by task type (supervisory vs focused)
- This is the SOSP figure. The data generates itself from production
  use — just needs analysis scripts.

### R2: Silent Degradation Measurement
The hard experiment. How often does eviction cause worse output
without the model knowing to re-request?
- Paired trials: same task with/without eviction
- Need to control for baseline hallucination rate
- Confidence-at-time-of-write vs correctness-of-output
- Relates to ADV-A-002 from Rikuy review

### R3: Anchor/Demotion with Yanantin Integration
**This is where pichay and yanantin merge.**
Eviction becomes demotion, not loss:
- L1: Context window (managed by proxy)
- L2: Yanantin episodic/semantic store (evicted content, indexed)
- L3: Raw JSONL logs (complete but unindexed)
Anchor = TLB entry. Summary + retrieval handle left in context.
On retrieval need, query Yanantin, promote back to L1.
- Search space bounded to session's eviction set (20-50 chunks)
- Temporal locality from proxy timestamps
- Spatial locality from Yanantin graph connections
- Directly addresses silent degradation (anchors make eviction
  visible to the model)

### R4: Fault Detection in Probabilistic Memory Systems
The paper after the paper. The VM analogy holds for management but
breaks for failure modes:
- Hardware: page fault is deterministic, CPU doesn't guess evicted
  page contents
- LLM: eviction produces distribution shift, model confabulates
  through gaps, "wrong because evicted" indistinguishable from
  "wrong because baseline hallucination"
- Cost of fault in hardware: known (disk IO). Cost of fault in LLM:
  unknown for any individual eviction.
- Research opportunity: formal model of fault detection where the
  memory consumer is a stochastic process

### R5: Multi-User Corpus
Paper's N=1 limitation. Get other Claude Code users running through
the proxy in observation mode. Compare working set profiles,
eviction rates, fault rates across users and task types.

## Ship Priorities

### S1: arXiv v1 (Context Waste Paper)
Paper at paper/main.tex on paper branch. Rikuy-reviewed, fixes
applied. Ready to submit. Tony's plan: merge experiments into main
on release day + README, push paper, update repo with arXiv bibtex.

### S2: "Replicate on Your Traces" Script
Let users run the analyzer on their own Claude Code logs and see
their waste profile. Strongest adoption driver. Exists partially
in tools/ but needs packaging.

### S3: README for Public Repo
Needed for release day. Cover: what pichay is, how to install, how
to run (observation vs compact vs trim modes), how to read the
dashboard, link to paper.

## Yanantin Integration Points

These are the items where pichay and yanantin need to coordinate:

1. **R3 (Anchors)**: Yanantin provides the L2 backing store. Needs
   an API for: write evicted content with summary key, query by
   summary, retrieve full content. Session-scoped.
2. **E1 (Conversation eviction)**: Evicted conversation turns could
   go to Yanantin as episodic memories. Retrieval on semantic match.
3. **R1 (Working set curve)**: Yanantin's query layer could surface
   historical working set data across sessions for longitudinal
   analysis.
4. **R5 (Multi-user)**: Yanantin could aggregate eviction profiles
   across users/sessions for corpus-scale analysis.

## Session Numbers for Reference

| Metric | Value |
|--------|-------|
| Session duration | 3+ hours |
| API calls | 202+ (main session) |
| Agents spawned | 5 |
| Byte reduction | 53% (main), 20-40% (agents) |
| Working set range | 44-69% of 200K cap |
| Evictions | 4,495 |
| Faults | 8 (0.18%) |
| Silent degradation | 1 confirmed (proxy.py API bug) |
| Papers edited | 2 (pichay, neutrosophic) |
| Rikuy reviews | 4 (2 per paper) |
| Experiments run | 2 (tautology ablation, prompt ablation) |
