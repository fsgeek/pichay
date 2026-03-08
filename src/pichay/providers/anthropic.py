from __future__ import annotations

from copy import deepcopy

from pichay.core.models import CanonicalMessage, CanonicalRequest


class AnthropicAdapter:
    name = "anthropic"

    def normalize_request(self, payload: dict) -> CanonicalRequest:
        payload = deepcopy(payload)
        messages: list[CanonicalMessage] = []
        for msg in payload.get("messages", []):
            content = msg.get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            messages.append(
                CanonicalMessage(
                    role=msg.get("role", "user"),
                    content=content,
                    raw=msg,
                )
            )

        extensions = {
            k: v for k, v in payload.items()
            if k not in {"model", "max_tokens", "stream", "messages", "tools", "system"}
        }

        return CanonicalRequest(
            provider=self.name,
            model=payload.get("model", ""),
            max_tokens=payload.get("max_tokens"),
            stream=bool(payload.get("stream", False)),
            messages=messages,
            tools=payload.get("tools", []) or [],
            system=payload.get("system"),
            extensions=extensions,
        )

    def denormalize_request(self, req: CanonicalRequest) -> dict:
        body = {
            "model": req.model,
            "stream": req.stream,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in req.messages
            ],
        }
        if req.max_tokens is not None:
            body["max_tokens"] = req.max_tokens
        if req.tools:
            body["tools"] = req.tools
        if req.system is not None:
            body["system"] = req.system
        body.update(req.extensions)
        return body

    def upstream_path(self, req: CanonicalRequest, endpoint: str) -> str:
        if endpoint == "count_tokens":
            return "/v1/messages/count_tokens"
        return "/v1/messages"
