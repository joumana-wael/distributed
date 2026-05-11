import time
import argparse
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from rag.retriever import retrieve_context
from llm.inference import run_llm


app = FastAPI(title="GPU Worker API")

worker_id = None
is_alive = True
completed_tasks = 0
failed_tasks = 0
active_connections = 0

state_lock = threading.Lock()


class RequestModel(BaseModel):
    id: int
    query: str


@app.get("/health")
def health():
    with state_lock:
        return {
            "worker_id": worker_id,
            "is_alive": is_alive,
            "active_connections": active_connections,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks
        }


@app.post("/fail")
def fail_worker():
    global is_alive

    with state_lock:
        is_alive = False

    print(f"[FAULT] Worker {worker_id} has FAILED")

    return {
        "worker_id": worker_id,
        "status": "failed"
    }


@app.post("/recover")
def recover_worker():
    global is_alive

    with state_lock:
        is_alive = True

    print(f"[RECOVERY] Worker {worker_id} has RECOVERED")

    return {
        "worker_id": worker_id,
        "status": "recovered"
    }


@app.post("/process")
def process_request(request: RequestModel):
    global active_connections, completed_tasks, failed_tasks

    with state_lock:
        if not is_alive:
            raise HTTPException(
                status_code=503,
                detail=f"Worker {worker_id} is down"
            )

        active_connections += 1

    start = time.time()

    try:
        print(f"[Worker {worker_id}] Processing request {request.id}")

        context = retrieve_context(request.query)

        print(f"[Worker {worker_id}] Retrieved context for request {request.id}:")
        print(context)

        llm_output = run_llm(request.query, context)

        if isinstance(llm_output, dict):
            result = llm_output.get("answer", "")
            gpu_utilization = llm_output.get("gpu_utilization", 0)
            inference_mode = llm_output.get("mode", "unknown")
        else:
            result = llm_output
            gpu_utilization = 0
            inference_mode = "legacy"

        with state_lock:
            if not is_alive:
                raise HTTPException(
                    status_code=503,
                    detail=f"Worker {worker_id} failed during execution"
                )

        end = time.time()
        latency = end - start

        with state_lock:
            completed_tasks += 1

        return {
            "id": request.id,
            "worker_id": worker_id,
            "query": request.query,
            "result": result,
            "start_time": start,
            "end_time": end,
            "latency": latency,
            "gpu_utilization": gpu_utilization,
            "inference_mode": inference_mode,
            "status": "success",
            "error": ""
        }

    except HTTPException:
        with state_lock:
            failed_tasks += 1

        raise

    except Exception as e:
        with state_lock:
            failed_tasks += 1

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        with state_lock:
            active_connections -= 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)

    args = parser.parse_args()

    worker_id = args.worker_id

    print(f"[Worker {worker_id}] Starting REST API on port {args.port}")

    uvicorn.run(app, host="0.0.0.0", port=args.port)