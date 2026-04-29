
import threading
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Common.models import Request


class ClientLoadGenerator:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def send_request(self, request_id):
        request = Request(request_id, f"Query {request_id}")
        response = self.scheduler.handle_request(request)
        print(f"Response {response.id}: {response.result} (Latency: {response.latency:.4f}s)")

    def run_load_test(self, num_users=1000):
        threads = []

        for i in range(num_users):
            t = threading.Thread(target=self.send_request, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
