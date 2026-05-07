import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from Workers.worker_client import WorkerClient
from lb.load_balancer import LoadBalancer
from Master.Scheduler import Scheduler


app = FastAPI(title="Master Node API")

scheduler = None
configured_workers = []


class RequestModel(BaseModel):
    id: int
    query: str


@app.get("/health")
def health():
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler is not initialized")

    return {
        "master_id": scheduler.master_id,
        "is_alive": scheduler.is_available(),
        "total_requests": scheduler.total_requests,
        "completed_requests": scheduler.completed_requests,
        "failed_requests": scheduler.failed_requests,
        "workers": configured_workers
    }


@app.post("/fail")
def fail_master():
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler is not initialized")

    scheduler.fail()

    return {
        "master_id": scheduler.master_id,
        "status": "failed"
    }


@app.post("/recover")
def recover_master():
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler is not initialized")

    scheduler.recover()

    return {
        "master_id": scheduler.master_id,
        "status": "recovered"
    }


@app.get("/summary")
def summary():
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler is not initialized")

    avg_latency = (
        scheduler.total_latency / scheduler.completed_requests
        if scheduler.completed_requests > 0
        else 0.0
    )

    return {
        "master_id": scheduler.master_id,
        "is_alive": scheduler.is_available(),
        "total_requests": scheduler.total_requests,
        "completed_requests": scheduler.completed_requests,
        "failed_requests": scheduler.failed_requests,
        "average_latency": avg_latency,
        "min_latency": scheduler.min_latency if scheduler.completed_requests > 0 else 0.0,
        "max_latency": scheduler.max_latency,
        "workers": configured_workers
    }


@app.post("/handle_request")
def handle_request(request: RequestModel):
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler is not initialized")

    if not scheduler.is_available():
        raise HTTPException(
            status_code=503,
            detail=f"Master {scheduler.master_id} is down"
        )

    try:
        response = scheduler.handle_request(request)
        return response

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


def build_local_workers(worker_start, worker_count, worker_port_start):
    workers = []
    worker_descriptions = []

    for i in range(worker_count):
        worker_id = worker_start + i
        port = worker_port_start + i
        url = f"http://127.0.0.1:{port}"

        workers.append(
            WorkerClient(
                worker_id,
                url
            )
        )

        worker_descriptions.append({
            "worker_id": worker_id,
            "url": url
        })

    return workers, worker_descriptions


def build_remote_workers_from_urls(worker_urls):
    """
    Expected format:
    "0=http://WORKER0_IP:8001,1=http://WORKER1_IP:8001"

    Example:
    --worker-urls "0=http://10.0.0.5:8001,1=http://10.0.0.6:8001"
    """

    workers = []
    worker_descriptions = []

    mappings = [item.strip() for item in worker_urls.split(",") if item.strip()]

    if not mappings:
        raise ValueError("worker-urls was provided but no valid worker mappings were found.")

    for mapping in mappings:
        if "=" not in mapping:
            raise ValueError(
                f"Invalid worker mapping: {mapping}. "
                "Expected format: worker_id=http://host:port"
            )

        worker_id_str, url = mapping.split("=", 1)

        worker_id = int(worker_id_str.strip())
        url = url.strip()

        workers.append(
            WorkerClient(
                worker_id,
                url
            )
        )

        worker_descriptions.append({
            "worker_id": worker_id,
            "url": url
        })

    return workers, worker_descriptions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--master-id", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)

    # Remote multi-instance mode
    parser.add_argument(
        "--worker-urls",
        type=str,
        required=False,
        default=None,
        help=(
            "Comma-separated worker mapping. "
            "Example: 0=http://10.0.0.5:8001,1=http://10.0.0.6:8001"
        )
    )

    # Local same-machine mode
    parser.add_argument("--worker-start", type=int, required=False)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--worker-port-start", type=int, required=False)

    parser.add_argument("--strategy", type=str, default="round_robin")

    args = parser.parse_args()

    if args.worker_urls:
        workers, configured_workers = build_remote_workers_from_urls(args.worker_urls)

        print(
            f"[Master {args.master_id}] Using REMOTE worker URLs:"
        )

        for worker in configured_workers:
            print(f"  Worker {worker['worker_id']} → {worker['url']}")

    else:
        if args.worker_start is None or args.worker_port_start is None:
            raise ValueError(
                "For local mode, you must provide --worker-start and --worker-port-start. "
                "For remote Thunder mode, provide --worker-urls."
            )

        workers, configured_workers = build_local_workers(
            worker_start=args.worker_start,
            worker_count=args.worker_count,
            worker_port_start=args.worker_port_start
        )

        print(
            f"[Master {args.master_id}] Using LOCAL workers "
            f"{args.worker_start} to {args.worker_start + args.worker_count - 1}"
        )

        for worker in configured_workers:
            print(f"  Worker {worker['worker_id']} → {worker['url']}")

    local_lb = LoadBalancer(workers, strategy=args.strategy)

    scheduler = Scheduler(local_lb, master_id=args.master_id)

    print(
        f"[Master {args.master_id}] Starting REST API on port {args.port} "
        f"using strategy={args.strategy}"
    )

    uvicorn.run(app, host="0.0.0.0", port=args.port)