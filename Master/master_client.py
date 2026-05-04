import requests


class MasterClient:
    def __init__(self, master_id, base_url):
        self.master_id = master_id
        self.id = master_id
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

    def handle_request(self, request):
        try:
            payload = {
                "id": request.id,
                "query": request.query
            }

            response = requests.post(
                f"{self.base_url}/handle_request",
                json=payload,
                timeout=180
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Master {self.master_id} API error: {response.text}"
                )

            data = response.json()

            if data.get("status", "success") == "failed":
                raise RuntimeError(
                    f"Master {self.master_id} failed request: {data.get('error', '')}"
                )

            return data

        except requests.RequestException as e:
            raise RuntimeError(f"Master {self.master_id} connection failed: {e}")

    def fail(self):
        response = requests.post(f"{self.base_url}/fail", timeout=2)
        return response.json()

    def recover(self):
        response = requests.post(f"{self.base_url}/recover", timeout=2)
        return response.json()

    def get_summary(self):
        response = requests.get(f"{self.base_url}/summary", timeout=5)
        return response.json()