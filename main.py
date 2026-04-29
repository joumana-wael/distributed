from Workers.gpu_worker import GPUWorker
from lb.load_balancer import LoadBalancer
from Master.Scheduler import Scheduler

def main():
    # Create GPU workers
    workers = [GPUWorker(i) for i in range(4)] #simulating 4 GPU workers

    # Create Load Balancer with GPU workers
    lb = LoadBalancer(workers)

    # Create Scheduler
    scheduler = Scheduler(lb)

    # Run Simulation
    run_load_test(scheduler, num_users=1000)

if __name__ == "__main__":
    main()