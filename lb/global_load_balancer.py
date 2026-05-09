import threading
import time


class GlobalLoadBalancer:
    def __init__(self, masters, strategy="global_round_robin"):
        self.masters = masters
        self.strategy = strategy
        self.index = 0
        self.lock = threading.Lock()

    def get_active_masters(self):
        return [
            master
            for master in self.masters
            if master.is_available()
        ]

    def get_next_master(self):
        max_retries = 3
        retry_delay = 0.2

        for attempt in range(max_retries):
            active_masters = self.get_active_masters()

            if len(active_masters) > 0:
                with self.lock:
                    master = active_masters[self.index % len(active_masters)]
                    self.index = (self.index + 1) % len(active_masters)

                return master

            print(
                f"[Global LB] No available masters detected. "
                f"Retrying health check {attempt + 1}/{max_retries}..."
            )

            time.sleep(retry_delay)

        raise RuntimeError("No available master nodes")

    def dispatch(self, request):
        tried_master_ids = set()
        last_error = None
        max_attempts = len(self.masters)

        for attempt in range(max_attempts):
            master = self.get_next_master()

            if master.master_id in tried_master_ids:
                continue

            tried_master_ids.add(master.master_id)

            try:
                print(
                    f"[Global LB] Sending request {request.id} "
                    f"to Master {master.master_id}"
                )

                return master.handle_request(request)

            except RuntimeError as e:
                last_error = e

                print(
                    f"[Global LB] Master {master.master_id} failed for request {request.id}. "
                    f"Reassigning request to another master..."
                )

        raise RuntimeError(
            f"Request {request.id} failed after master reassignment attempts: {last_error}"
        )