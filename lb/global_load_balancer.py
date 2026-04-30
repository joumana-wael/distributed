import threading


class GlobalLoadBalancer:
    def __init__(self, masters, strategy="global_round_robin"):
        self.masters = masters
        self.strategy = strategy
        self.index = 0
        self.lock = threading.Lock()

    def get_active_masters(self):
        return [master for master in self.masters if master.is_available()]

    def get_next_master(self):
        with self.lock:
            active_masters = self.get_active_masters()

            if len(active_masters) == 0:
                raise RuntimeError("No available master nodes")

            master = active_masters[self.index % len(active_masters)]
            self.index = (self.index + 1) % len(active_masters)

            return master

    def dispatch(self, request):
        max_attempts = len(self.masters)
        last_error = None

        for attempt in range(max_attempts):
            master = self.get_next_master()

            try:
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