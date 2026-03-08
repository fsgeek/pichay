from __future__ import annotations

from copy import deepcopy

from pichay.core.models import CanonicalMessage, CanonicalRequest


class OpenAIAdapter:
    name = "openai"

    def normalize_request(self, payload: dict) -> CanonicalRequest:
        payload = deepcopy(payload)
        messages: list[CanonicalMessage] = []
        for msg in payload.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str):
                blocks = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                blocks = []
                for item in content:
                    if isinstance(item, str):
                        blocks.append({"type": "text", "text": item})
                    elif isinstance(item, dict):
                        if item.get("type") == "text" and "text" in item:
                            blocks.append(item)
                        elif item.get("type") == "input_text" and "text" in item:
                            blocks.append({"type": "text", "text": item["text"]})
                        else:
                            blocks.append(item)
                    else:
                        blocks.append({"type": "text", "text": str(item)})
            else:
                blocks = [{"type": "text", "text": str(content)}]
            messages.append(CanonicalMessage(role=msg.get("role", "user"), content=blocks, raw=msg))

        extensions = {
            k: v for k, v in payload.items()
            if k not in {"model", "max_tokens", "stream", "messages", "tools", "response_format"}
        }
        if "response_format" in payload:
            extensions["response_format"] = payload["response_format"]

        return CanonicalRequest(
            provider=self.name,
            model=payload.get("model", ""),
            max_tokens=payload.get("max_tokens"),
            stream=bool(payload.get("stream", False)),
            messages=messages,
            tools=payload.get("tools", []) or [],
            extensions=extensions,
        )

    def denormalize_request(self, req: CanonicalRequest) -> dict:
        def to_openai_content(blocks: list[dict]) -> str | list[dict]:
            if all(b.get("type") == "text" and isinstance(b.get("text"), str) for b in blocks):
                # Preserve compatibility with classic chat completions.
                return "\n".join(b.get("text", "") for b in blocks)
            return blocks

        body = {
            "model": req.model,
            "stream": req.stream,
            "messages": [
                {"role": m.role, "content": to_openai_content(m.content)}
                for m in req.messages
            ],
        }
        if req.max_tokens is not None:
            body["max_tokens"] = req.max_tokens
        if req.tools:
            body["tools"] = req.tools
        body.update(req.extensions)
        return body

    def upstream_path(self, req: CanonicalRequest, endpoint: str) -> str:
        return "/v1/chat/completions"
