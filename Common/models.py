from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict


@dataclass
class Request:
    request_id: str
    query: str
    created_at: float = field(default_factory=perf_counter)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    request_id: str
    worker_id: str
    answer: str
    latency_ms: float
    success: bool = True
    error: str = ""

