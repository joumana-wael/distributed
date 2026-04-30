import time
import threading
from rag.retriever import retrieve_context
from llm.inference import run_llm


class GPUWorker:
    def __init__(self, id):
        self.id = id

        # Fault tolerance status
        self.is_alive = True

        # Metrics
        self.active_connections = 0
        self.completed_tasks = 0
        self.failed_tasks = 0

        self.lock = threading.Lock()

    def fail(self):
        with self.lock:
            self.is_alive = False
        print(f"[FAULT] Worker {self.id} has FAILED")

    def recover(self):
        with self.lock:
            self.is_alive = True
        print(f"[RECOVERY] Worker {self.id} has RECOVERED")

    def is_available(self):
        with self.lock:
            return self.is_alive

    def get_active_connections(self):
        with self.lock:
            return self.active_connections

    def process(self, request):
        if not self.is_available():
            raise RuntimeError(f"Worker {self.id} is down")

        with self.lock:
            self.active_connections += 1

        start = time.time()

        try:
            print(f"[Worker {self.id}] Processing request {request.id}")

            context = retrieve_context(request.query)

            if not self.is_available():
                raise RuntimeError(f"Worker {self.id} failed before inference")

            result = run_llm(request.query, context)

            if not self.is_available():
                raise RuntimeError(f"Worker {self.id} failed during execution")

            end = time.time()
            latency = end - start

            with self.lock:
                self.completed_tasks += 1

            return {
                "id": request.id,
                "worker_id": self.id,
                "query": request.query,
                "result": result,
                "start_time": start,
                "end_time": end,
                "latency": latency,
                "status": "success",
                "error": ""
            }

        except RuntimeError as e:
            with self.lock:
                self.failed_tasks += 1

            print(f"[Worker {self.id}] ERROR while processing request {request.id}: {e}")
            raise e

        finally:
            with self.lock:
                self.active_connections -= 1