import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from Workers.worker_client import WorkerClient
from lb.load_balancer import LoadBalancer
from Master.Scheduler import Scheduler


app = FastAPI()

scheduler = None


class RequestModel(BaseModel):
    id: int
    query: str


@app.get("/health")
def health():
    return {
        "master_id": scheduler.master_id,
        "is_alive": scheduler.is_available(),
        "total_requests": scheduler.total_requests,
        "completed_requests": scheduler.completed_requests,
        "failed_requests": scheduler.failed_requests
    }


@app.post("/fail")
def fail_master():
    scheduler.fail()
    return {
        "master_id": scheduler.master_id,
        "status": "failed"
    }


@app.post("/recover")
def recover_master():
    scheduler.recover()
    return {
        "master_id": scheduler.master_id,
        "status": "recovered"
    }


@app.get("/summary")
def summary():
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
        "max_latency": scheduler.max_latency
    }


@app.post("/handle_request")
def handle_request(request: RequestModel):
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


def build_workers(worker_start, worker_count, worker_port_start):
    workers = []

    for i in range(worker_count):
        worker_id = worker_start + i
        port = worker_port_start + i

        workers.append(
            WorkerClient(
                worker_id,
                f"http://127.0.0.1:{port}"
            )
        )

    return workers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--master-id", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)

    parser.add_argument("--worker-start", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--worker-port-start", type=int, required=True)

    parser.add_argument("--strategy", type=str, default="round_robin")

    args = parser.parse_args()

    workers = build_workers(
        worker_start=args.worker_start,
        worker_count=args.worker_count,
        worker_port_start=args.worker_port_start
    )

    local_lb = LoadBalancer(workers, strategy=args.strategy)

    scheduler = Scheduler(local_lb, master_id=args.master_id)

    print(
        f"[Master {args.master_id}] Starting REST API on port {args.port} "
        f"with workers {args.worker_start} to {args.worker_start + args.worker_count - 1}"
    )

    uvicorn.run(app, host="127.0.0.1", port=args.port)