import threading
import time

from Workers.gpu_worker import GPUWorker
from lb.load_balancer import LoadBalancer
from lb.global_load_balancer import GlobalLoadBalancer
from Master.Scheduler import Scheduler
from client.load_generator import run_load_test
from llm.inference import preload_model


def fail_worker_later(worker, delay):
    time.sleep(delay)
    worker.fail()


def fail_master_later(master, delay):
    time.sleep(delay)
    master.fail()


def print_worker_pool_status(pool_name, workers):
    print(f"\n========== {pool_name} STATUS ==========")

    for worker in workers:
        print(
            f"Worker {worker.id} | "
            f"Alive: {worker.is_available()} | "
            f"Completed: {worker.completed_tasks} | "
            f"Failed while processing: {worker.failed_tasks}"
        )

    print("========================================\n")


def main():
    # Worker Pool A for Master 1
    workers_a = [GPUWorker(i) for i in range(4)]       # Workers 0,1,2,3

    # Worker Pool B for Master 2
    workers_b = [GPUWorker(i) for i in range(4, 8)]    # Workers 4,5,6,7

    # Local load balancers
    lb1 = LoadBalancer(workers_a, strategy="round_robin")
    lb2 = LoadBalancer(workers_b, strategy="round_robin")

    # Master nodes
    master1 = Scheduler(lb1, master_id=1)
    master2 = Scheduler(lb2, master_id=2)

    # Global load balancer
    global_lb = GlobalLoadBalancer(
        masters=[master1, master2],
        strategy="global_round_robin"
    )

    # Keep these defined, but do not start them while testing the real model
    worker_failure_1 = threading.Thread(
        target=fail_worker_later,
        args=(workers_a[2], 0.10)
    )

    worker_failure_2 = threading.Thread(
        target=fail_worker_later,
        args=(workers_b[2], 0.15)
    )

    master_failure = threading.Thread(
        target=fail_master_later,
        args=(master1, 0.05)
    )

    # REAL LLM TEST MODE:
    # Do NOT start fault tolerance threads here.
    # worker_failure_1.start()
    # worker_failure_2.start()
    # master_failure.start()

    # Load model once before starting client threads
    preload_model()

    # Start with 1 user only. After it works, try 5.
    run_load_test(global_lb, num_users=1000)

    # Do NOT join fault threads because they were not started.
    # worker_failure_1.join()
    # worker_failure_2.join()
    # master_failure.join()

    master1.print_master_summary()
    master2.print_master_summary()

    print_worker_pool_status("WORKER POOL A - MASTER 1", workers_a)
    print_worker_pool_status("WORKER POOL B - MASTER 2", workers_b)


if __name__ == "__main__":
    main()