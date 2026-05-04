import requests


class WorkerClient:
    def __init__(self, worker_id, base_url):
        self.id = worker_id
        self.base_url = base_url

    def is_available(self):
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)

            if response.status_code != 200:
                return False

            data = response.json()
            return data.get("is_alive", False)

        except requests.RequestException:
            return False

    def get_active_connections(self):
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)

            if response.status_code != 200:
                return 999999

            data = response.json()
            return data.get("active_connections", 999999)

        except requests.RequestException:
            return 999999

    def get_load_score(self):
        return self.get_active_connections()

    def process(self, request):
        try:
            payload = {
                "id": request.id,
                "query": request.query
            }

            response = requests.post(
                f"{self.base_url}/process",
                json=payload,
                timeout=120
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Worker {self.id} API error: {response.text}"
                )

            return response.json()

        except requests.RequestException as e:
            raise RuntimeError(f"Worker {self.id} connection failed: {e}")

    def fail(self):
        response = requests.post(f"{self.base_url}/fail", timeout=2)
        return response.json()

    def recover(self):
        response = requests.post(f"{self.base_url}/recover", timeout=2)
        return response.json()