from Master.master_client import MasterClient
from lb.global_load_balancer import GlobalLoadBalancer
from client.load_generator import run_load_test


def print_master_summary(master_client):
    try:
        summary = master_client.get_summary()

        print(f"\n========== MASTER {summary['master_id']} SUMMARY ==========")
        print(f"Alive               : {summary['is_alive']}")
        print(f"Total requests      : {summary['total_requests']}")
        print(f"Completed requests  : {summary['completed_requests']}")
        print(f"Failed requests     : {summary['failed_requests']}")
        print(f"Average latency     : {summary['average_latency']:.4f} seconds")
        print(f"Minimum latency     : {summary['min_latency']:.4f} seconds")
        print(f"Maximum latency     : {summary['max_latency']:.4f} seconds")
        print("========================================\n")

    except Exception as e:
        print(f"Could not read summary for Master {master_client.master_id}: {e}")


def main():
    # These are REST clients that communicate with the running Master APIs
    master1 = MasterClient(
        master_id=1,
        base_url="http://127.0.0.1:9001"
    )

    master2 = MasterClient(
        master_id=2,
        base_url="http://127.0.0.1:9002"
    )

    # Global load balancer distributes requests to Master REST APIs
    global_lb = GlobalLoadBalancer(
        masters=[master1, master2],
        strategy="global_round_robin"
    )

    #for users in [500, 1000]:
        #print(f"\n\n===== RUNNING LOAD TEST: {users} USERS =====")
    run_load_test(global_lb, num_users=20)

    print_master_summary(master1)
    print_master_summary(master2)


if __name__ == "__main__":
    main()