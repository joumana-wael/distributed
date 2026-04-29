# master/scheduler.py

import time
import threading
from common.models import Request, Response

class Scheduler:
    def __init__(self, load_balancer):
        self.lb = load_balancer
        self.lock = threading.Lock()
        self.total_requests     = 0
        self.completed_requests = 0
        self.failed_requests    = 0
        self.total_latency      = 0.0
        self.min_latency        = float('inf')
        self.max_latency        = 0.0
        self.start_time         = time.time()

    def handle_request(self, request: Request) -> Response:
        with self.lock:
            self.total_requests += 1
        print(f"[Scheduler] Dispatching request {request.id} → Load Balancer")
        try:
            response = self.lb.dispatch(request)
            with self.lock:
                self.completed_requests += 1
                self.total_latency      += response.latency
                self.min_latency = min(self.min_latency, response.latency)
                self.max_latency = max(self.max_latency, response.latency)
            return response
        except RuntimeError as e:
            with self.lock:
                self.failed_requests += 1
            print(f"[Scheduler] ERROR: Request {request.id} FAILED — {e}")
            return Response(
                id=request.id,
                result="ERROR: No available workers",
                latency=0.0,
                worker_id=-1
            )

    def get_metrics(self) -> dict:
        with self.lock:
            elapsed_time = time.time() - self.start_time
            avg_latency  = (
                self.total_latency / self.completed_requests
                if self.completed_requests > 0 else 0.0
            )
            throughput = (
                self.completed_requests / elapsed_time
                if elapsed_time > 0 else 0.0
            )
            return {
                "total_requests":   self.total_requests,
                "completed":        self.completed_requests,
                "failed":           self.failed_requests,
                "elapsed_time_sec": round(elapsed_time, 2),
                "avg_latency_sec":  round(avg_latency, 4),
                "min_latency_sec":  round(self.min_latency, 4),
                "max_latency_sec":  round(self.max_latency, 4),
                "throughput_rps":   round(throughput, 2),
            }