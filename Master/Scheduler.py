import threading
from Common.models import Request, Response


class Scheduler:
    def __init__(self, lb, master_id=0):
        self.lb = lb
        self.master_id = master_id

        # Master node health status
        self.is_alive = True

        self.lock = threading.Lock()

        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0

        self.total_latency = 0.0
        self.min_latency = float("inf")
        self.max_latency = 0.0

    def fail(self):
        with self.lock:
            self.is_alive = False
        print(f"[FAULT] Master {self.master_id} has FAILED")

    def recover(self):
        with self.lock:
            self.is_alive = True
        print(f"[RECOVERY] Master {self.master_id} has RECOVERED")

    def is_available(self):
        with self.lock:
            return self.is_alive

    def handle_request(self, request: Request) -> Response:
        if not self.is_available():
            raise RuntimeError(f"Master {self.master_id} is down")

        with self.lock:
            self.total_requests += 1

        print(
            f"[Master {self.master_id}] Received request {request.id} "
            f"→ Local Load Balancer"
        )

        try:
            response = self.lb.dispatch(request)

            if isinstance(response, dict):
                latency = response.get("latency", 0.0)
                response["master_id"] = self.master_id
            else:
                latency = response.latency

            with self.lock:
                self.completed_requests += 1
                self.total_latency += latency
                self.min_latency = min(self.min_latency, latency)
                self.max_latency = max(self.max_latency, latency)

            return response

        except RuntimeError as e:
            with self.lock:
                self.failed_requests += 1

            print(
                f"[Master {self.master_id}] ERROR: "
                f"Request {request.id} FAILED — {e}"
            )

            raise e

    def print_master_summary(self):
        with self.lock:
            avg_latency = (
                self.total_latency / self.completed_requests
                if self.completed_requests > 0
                else 0
            )

            print(f"\n========== MASTER {self.master_id} SUMMARY ==========")
            print(f"Alive               : {self.is_alive}")
            print(f"Total requests      : {self.total_requests}")
            print(f"Completed requests  : {self.completed_requests}")
            print(f"Failed requests     : {self.failed_requests}")
            print(f"Average latency     : {avg_latency:.4f} seconds")

            if self.completed_requests > 0:
                print(f"Minimum latency     : {self.min_latency:.4f} seconds")
                print(f"Maximum latency     : {self.max_latency:.4f} seconds")

            print("========================================\n")