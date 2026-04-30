import threading
import csv
import time
import os
from Common.models import Request


results = []
results_lock = threading.Lock()

def get_value(response, key, default=None):
    if isinstance(response, dict):
        return response.get(key, default)

    return getattr(response, key, default)

def save_results_to_csv(filename):
    os.makedirs("logs", exist_ok=True)

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "request_id",
            "master_id",
            "worker_id",
            "query",
            "start_time",
            "end_time",
            "latency",
            "status",
            "error"
        ])

        for r in results:
            writer.writerow([
                get_value(r, "id"),
                get_value(r, "master_id"),
                get_value(r, "worker_id"),
                get_value(r, "query"),
                get_value(r, "start_time"),
                get_value(r, "end_time"),
                get_value(r, "latency"),
                get_value(r, "status", "success"),
                get_value(r, "error", "")
            ])


def simulate_user(entry_point, user_id):
    try:
        request = Request(user_id, f"Query {user_id}")

        if hasattr(entry_point, "handle_request"):
            response = entry_point.handle_request(request)
        else:
            response = entry_point.dispatch(request)

        with results_lock:
            results.append(response)

        status = get_value(response, "status", "success")

        if status == "success":
            print(f"[Client] Request {user_id} completed")
        else:
            print(f"[Client] Request {user_id} failed")

    except Exception as e:
        with results_lock:
            results.append({
                "id": user_id,
                "master_id": "N/A",
                "worker_id": "N/A",
                "query": f"Query {user_id}",
                "start_time": 0,
                "end_time": 0,
                "latency": 0,
                "status": "failed",
                "error": str(e)
            })

        print(f"[Client] Request {user_id} failed: {e}")


def print_summary(total_duration, strategy):
    total_requests = len(results)

    successful = [
        r for r in results
        if get_value(r, "status", "success") == "success"
    ]

    failed = total_requests - len(successful)

    latencies = [
        float(get_value(r, "latency", 0))
        for r in successful
        if float(get_value(r, "latency", 0)) > 0
    ]

    print(f"\n========== {strategy.upper()} FAULT TOLERANCE TEST SUMMARY ==========")
    print(f"Total requests       : {total_requests}")
    print(f"Successful requests  : {len(successful)}")
    print(f"Failed requests      : {failed}")

    if len(latencies) == 0:
        print("No successful latency values found.")
        print("Check Scheduler.py / Worker response format.")
        print("====================================================\n")
        return

    print(f"Average latency      : {sum(latencies) / len(latencies):.4f} seconds")
    print(f"Minimum latency      : {min(latencies):.4f} seconds")
    print(f"Maximum latency      : {max(latencies):.4f} seconds")
    print(f"Throughput           : {len(successful) / total_duration:.2f} requests/second")

    worker_counts = {}

    for r in successful:
        worker_id = get_value(r, "worker_id", "Unknown")
        worker_counts[worker_id] = worker_counts.get(worker_id, 0) + 1

    print("\nRequests per worker:")
    for worker_id, count in sorted(worker_counts.items()):
        print(f"Worker {worker_id}: {count} requests")

    print("====================================================\n")

def get_strategy_name(entry_point):
    # Case 1: old architecture, entry_point is Scheduler
    if hasattr(entry_point, "lb"):
        return getattr(entry_point.lb, "strategy", "unknown")

    # Case 2: new architecture, entry_point is GlobalLoadBalancer
    if hasattr(entry_point, "strategy"):
        return getattr(entry_point, "strategy", "global_round_robin")

    return "unknown"
def run_load_test(scheduler, num_users=100):
    global results
    results = []

    strategy = get_strategy_name(scheduler)

    threads = []
    test_start = time.time()

    for i in range(num_users):
        thread = threading.Thread(target=simulate_user, args=(scheduler, i))
        threads.append(thread)
        thread.start()
        time.sleep(0.001)

    for thread in threads:
        thread.join()

    test_end = time.time()
    total_duration = test_end - test_start

    filename = f"logs/results_{strategy}_fault_tolerance_{num_users}_users.csv"

    save_results_to_csv(filename)
    print_summary(total_duration, strategy)

    print(f"CSV file saved at: {filename}")