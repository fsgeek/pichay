from __future__ import annotations

from typing import Protocol

from pichay.core.models import CanonicalRequest


class ProviderAdapter(Protocol):
    name: str

    def normalize_request(self, payload: dict) -> CanonicalRequest: ...

    def denormalize_request(self, req: CanonicalRequest) -> dict: ...

    def upstream_path(self, req: CanonicalRequest, endpoint: str) -> str: ...
