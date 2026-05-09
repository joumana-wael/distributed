import threading


class LoadBalancer:
    def __init__(self, workers, strategy="round_robin"):
        self.workers = workers
        self.strategy = strategy
        self.index = 0
        self.lock = threading.Lock()

    def get_active_workers(self, excluded_worker_ids=None):
        if excluded_worker_ids is None:
            excluded_worker_ids = set()

        return [
            worker
            for worker in self.workers
            if worker.id not in excluded_worker_ids and worker.is_available()
        ]

    def get_next_worker_round_robin(self, excluded_worker_ids=None):
        with self.lock:
            active_workers = self.get_active_workers(excluded_worker_ids)

            if len(active_workers) == 0:
                raise RuntimeError("No available workers")

            worker = active_workers[self.index % len(active_workers)]
            self.index = (self.index + 1) % len(active_workers)

            return worker

    def get_next_worker_least_connections(self, excluded_worker_ids=None):
        active_workers = self.get_active_workers(excluded_worker_ids)

        if len(active_workers) == 0:
            raise RuntimeError("No available workers")

        return min(
            active_workers,
            key=lambda worker: worker.get_active_connections()
        )

    def get_next_worker_load_aware(self, excluded_worker_ids=None):
        active_workers = self.get_active_workers(excluded_worker_ids)

        if len(active_workers) == 0:
            raise RuntimeError("No available workers")

        return min(
            active_workers,
            key=lambda worker: worker.get_load_score()
        )

    def get_next_worker(self, excluded_worker_ids=None):
        if self.strategy == "round_robin":
            return self.get_next_worker_round_robin(excluded_worker_ids)

        elif self.strategy == "least_connections":
            return self.get_next_worker_least_connections(excluded_worker_ids)

        elif self.strategy == "load_aware":
            return self.get_next_worker_load_aware(excluded_worker_ids)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def dispatch(self, request):
        tried_worker_ids = set()
        last_error = None

        max_attempts = len(self.workers)

        for attempt in range(max_attempts):
            try:
                worker = self.get_next_worker(tried_worker_ids)

            except RuntimeError:
                break

            tried_worker_ids.add(worker.id)

            try:
                print(
                    f"[Load Balancer] Sending request {request.id} "
                    f"to Worker {worker.id}"
                )

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