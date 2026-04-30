import threading


class LoadBalancer:
    def __init__(self, workers, strategy="round_robin"):
        self.workers = workers
        self.strategy = strategy
        self.index = 0
        self.lock = threading.Lock()

    def get_active_workers(self):
        return [worker for worker in self.workers if worker.is_available()]

    def get_next_worker_round_robin(self):
        with self.lock:
            active_workers = self.get_active_workers()

            if len(active_workers) == 0:
                raise RuntimeError("No available workers")

            worker = active_workers[self.index % len(active_workers)]
            self.index = (self.index + 1) % len(active_workers)

            return worker

    def get_next_worker_least_connections(self):
        with self.lock:
            active_workers = self.get_active_workers()

            if len(active_workers) == 0:
                raise RuntimeError("No available workers")

            return min(
                active_workers,
                key=lambda worker: worker.get_active_connections()
            )

    def get_next_worker_load_aware(self):
        with self.lock:
            active_workers = self.get_active_workers()

            if len(active_workers) == 0:
                raise RuntimeError("No available workers")

            return min(
                active_workers,
                key=lambda worker: worker.get_load_score()
            )

    def get_next_worker(self):
        if self.strategy == "round_robin":
            return self.get_next_worker_round_robin()

        elif self.strategy == "least_connections":
            return self.get_next_worker_least_connections()

        elif self.strategy == "load_aware":
            return self.get_next_worker_load_aware()

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def dispatch(self, request):
        max_attempts = len(self.workers)
        last_error = None

        for attempt in range(max_attempts):
            worker = self.get_next_worker()

            try:
                return worker.process(request)

            except RuntimeError as e:
                last_error = e

                print(
                    f"[Load Balancer] Worker {worker.id} failed for request {request.id}. "
                    f"Reassigning task..."
                )

        raise RuntimeError(
            f"Request {request.id} failed after reassignment attempts: {last_error}"
        )