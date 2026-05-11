import argparse
import csv
import os
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from Common.models import Request
from lb.load_balancer import LoadBalancer
from lb.global_load_balancer import GlobalLoadBalancer


class SimulatedGPUWorker:
    def __init__(self, worker_id, min_delay=0.05, max_delay=0.20):
        self.id = worker_id
        self.alive = True
        self.active_connections = 0
        self.completed_tasks = 0
        self.failed_tasks = 0

        self.min_delay = min_delay
        self.max_delay = max_delay

        self.lock = threading.Lock()

    def is_available(self):
        with self.lock:
            return self.alive

    def get_active_connections(self):
        with self.lock:
            return self.active_connections

    def get_load_score(self):
        with self.lock:
            return self.active_connections

    def fail(self):
        with self.lock:
            self.alive = False
        print(f"[FAULT] Worker {self.id} failed")

    def recover(self):
        with self.lock:
            self.alive = True
        print(f"[RECOVERY] Worker {self.id} recovered")

    def process(self, request):
        if not self.is_available():
            raise RuntimeError(f"Worker {self.id} is unavailable")

        start_time = time.time()

        with self.lock:
            self.active_connections += 1

        try:
            # Simulated LLM/GPU inference delay
            simulated_delay = random.uniform(self.min_delay, self.max_delay)
            time.sleep(simulated_delay)

            end_time = time.time()
            latency = end_time - start_time

            with self.lock:
                self.completed_tasks += 1

            return {
                "id": request.id,
                "worker_id": self.id,
                "query": request.query,
                "result": "SIMULATED_LLM_RESPONSE",
                "start_time": start_time,
                "end_time": end_time,
                "latency": latency,
                "status": "success",
                "error": ""
            }

        except Exception as e:
            with self.lock:
                self.failed_tasks += 1

            raise RuntimeError(str(e))

        finally:
            with self.lock:
                self.active_connections -= 1


class SimulatedMaster:
    def __init__(self, master_id, lb):
        self.master_id = master_id
        self.id = master_id
        self.lb = lb

        self.alive = True
        self.lock = threading.Lock()

        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0

        self.total_latency = 0.0
        self.min_latency = float("inf")
        self.max_latency = 0.0

    def is_available(self):
        with self.lock:
            return self.alive

    def fail(self):
        with self.lock:
            self.alive = False
        print(f"[FAULT] Master {self.master_id} failed")

    def recover(self):
        with self.lock:
            self.alive = True
        print(f"[RECOVERY] Master {self.master_id} recovered")

    def handle_request(self, request):
        if not self.is_available():
            raise RuntimeError(f"Master {self.master_id} is down")

        with self.lock:
            self.total_requests += 1

        try:
            response = self.lb.dispatch(request)
            latency = response.get("latency", 0.0)

            with self.lock:
                self.completed_requests += 1
                self.total_latency += latency
                self.min_latency = min(self.min_latency, latency)
                self.max_latency = max(self.max_latency, latency)

            response["master_id"] = self.master_id
            return response

        except RuntimeError as e:
            with self.lock:
                self.failed_requests += 1

            raise RuntimeError(str(e))


def simulate_user(global_lb, user_id, query, start_barrier):
    start_barrier.wait()

    request = Request(
        id=user_id,
        query=query
    )

    try:
        response = global_lb.dispatch(request)
        return response

    except Exception as e:
        return {
            "id": user_id,
            "worker_id": "N/A",
            "master_id": "N/A",
            "query": query,
            "result": "",
            "start_time": 0,
            "end_time": 0,
            "latency": 0,
            "status": "failed",
            "error": str(e)
        }


def save_results_to_csv(results, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fieldnames = [
        "id",
        "master_id",
        "worker_id",
        "query",
        "result",
        "start_time",
        "end_time",
        "latency",
        "status",
        "error"
    ]

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow({
                "id": result.get("id"),
                "master_id": result.get("master_id"),
                "worker_id": result.get("worker_id"),
                "query": result.get("query"),
                "result": result.get("result"),
                "start_time": result.get("start_time"),
                "end_time": result.get("end_time"),
                "latency": result.get("latency"),
                "status": result.get("status", "success"),
                "error": result.get("error", "")
            })


def print_summary(results, total_duration, users):
    successful = [r for r in results if r.get("status", "success") == "success"]
    failed = [r for r in results if r.get("status", "success") != "success"]

    latencies = [
        float(r.get("latency", 0))
        for r in successful
        if float(r.get("latency", 0)) > 0
    ]

    print(f"\n========== SIMULATED LOAD TEST SUMMARY: {users} USERS ==========")
    print(f"Total requests       : {len(results)}")
    print(f"Successful requests  : {len(successful)}")
    print(f"Failed requests      : {len(failed)}")
    print(f"Total duration       : {total_duration:.4f} seconds")

    if successful:
        print(f"Throughput           : {len(successful) / total_duration:.2f} requests/second")

    if latencies:
        print(f"Average latency      : {statistics.mean(latencies):.4f} seconds")
        print(f"Minimum latency      : {min(latencies):.4f} seconds")
        print(f"Maximum latency      : {max(latencies):.4f} seconds")

    master_counts = {}
    worker_counts = {}

    for r in successful:
        master_id = r.get("master_id", "N/A")
        worker_id = r.get("worker_id", "N/A")

        master_counts[master_id] = master_counts.get(master_id, 0) + 1
        worker_counts[worker_id] = worker_counts.get(worker_id, 0) + 1

    print("\nRequests per master:")
    for master_id, count in sorted(master_counts.items(), key=lambda x: str(x[0])):
        print(f"Master {master_id}: {count}")

    print("\nRequests per worker:")
    for worker_id, count in sorted(worker_counts.items(), key=lambda x: str(x[0])):
        print(f"Worker {worker_id}: {count}")

    if failed:
        print("\nSample errors:")
        for r in failed[:5]:
            print(f"Request {r.get('id')} failed: {r.get('error')}")

    print("====================================================\n")

    return {
        "users": users,
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "total_duration": total_duration,
        "throughput": len(successful) / total_duration if total_duration > 0 else 0,
        "average_latency": statistics.mean(latencies) if latencies else 0,
        "min_latency": min(latencies) if latencies else 0,
        "max_latency": max(latencies) if latencies else 0,
    }


def save_summary_csv(summaries, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fieldnames = [
        "users",
        "total_requests",
        "successful_requests",
        "failed_requests",
        "total_duration",
        "throughput",
        "average_latency",
        "min_latency",
        "max_latency",
        "csv_file"
    ]

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def build_simulated_system(strategy, min_delay, max_delay):
    # Master 1 workers
    workers_a = [
        SimulatedGPUWorker(0, min_delay, max_delay),
        SimulatedGPUWorker(1, min_delay, max_delay)
    ]

    # Master 2 workers
    workers_b = [
        SimulatedGPUWorker(2, min_delay, max_delay),
        SimulatedGPUWorker(3, min_delay, max_delay)
    ]

    lb1 = LoadBalancer(workers_a, strategy=strategy)
    lb2 = LoadBalancer(workers_b, strategy=strategy)

    master1 = SimulatedMaster(master_id=1, lb=lb1)
    master2 = SimulatedMaster(master_id=2, lb=lb2)

    global_lb = GlobalLoadBalancer(
        masters=[master1, master2],
        strategy="global_round_robin"
    )

    return global_lb, master1, master2, workers_a + workers_b


def run_load_test(users, strategy, query, min_delay, max_delay, output_dir):
    global_lb, master1, master2, workers = build_simulated_system(
        strategy=strategy,
        min_delay=min_delay,
        max_delay=max_delay
    )

    print(f"\nStarting simulated load test with {users} concurrent users...")
    print(f"Strategy: {strategy}")
    print(f"Simulated inference delay: {min_delay}s to {max_delay}s")

    start_barrier = threading.Barrier(users)
    results = []

    test_start = time.time()

    with ThreadPoolExecutor(max_workers=users) as executor:
        futures = []

        for i in range(users):
            future = executor.submit(
                simulate_user,
                global_lb,
                i,
                query,
                start_barrier
            )
            futures.append(future)

        for future in as_completed(futures):
            results.append(future.result())

            if len(results) % 100 == 0:
                print(f"Completed {len(results)}/{users} requests...")

    test_end = time.time()
    total_duration = test_end - test_start

    results.sort(key=lambda r: r.get("id", 0))

    csv_file = os.path.join(
        output_dir,
        f"simulated_load_test_{strategy}_{users}_users.csv"
    )

    save_results_to_csv(results, csv_file)

    summary = print_summary(results, total_duration, users)
    summary["csv_file"] = csv_file

    print(f"CSV saved at: {csv_file}")

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[100, 250, 500, 1000],
        help="Concurrent user levels to simulate"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="round_robin",
        choices=["round_robin", "least_connections", "load_aware"],
        help="Local worker load balancing strategy"
    )

    parser.add_argument(
        "--query",
        type=str,
        default="Simulated LLM inference request",
        help="Simulated query text"
    )

    parser.add_argument(
        "--min-delay",
        type=float,
        default=0.05,
        help="Minimum simulated inference delay in seconds"
    )

    parser.add_argument(
        "--max-delay",
        type=float,
        default=0.20,
        help="Maximum simulated inference delay in seconds"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/simulated_load_tests",
        help="Output folder for CSV files"
    )

    args = parser.parse_args()

    summaries = []

    for users in args.levels:
        summary = run_load_test(
            users=users,
            strategy=args.strategy,
            query=args.query,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
            output_dir=args.output_dir
        )

        summaries.append(summary)
        time.sleep(2)

    summary_file = os.path.join(
        args.output_dir,
        f"simulated_load_test_summary_{args.strategy}.csv"
    )

    save_summary_csv(summaries, summary_file)

    print("\nAll simulated load tests completed.")
    print(f"Summary CSV saved at: {summary_file}")


if __name__ == "__main__":
    main()