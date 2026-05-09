import time
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from rag.retriever import retrieve_context
from llm.inference import run_llm


app = FastAPI()

worker_id = None
is_alive = True
completed_tasks = 0
failed_tasks = 0
active_connections = 0


class RequestModel(BaseModel):
    id: int
    query: str


@app.get("/health")
def health():
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
    is_alive = False
    return {
        "worker_id": worker_id,
        "status": "failed"
    }


@app.post("/recover")
def recover_worker():
    global is_alive
    is_alive = True
    return {
        "worker_id": worker_id,
        "status": "recovered"
    }


@app.post("/process")
def process_request(request: RequestModel):
    global active_connections, completed_tasks, failed_tasks

    if not is_alive:
        raise HTTPException(status_code=503, detail=f"Worker {worker_id} is down")

    active_connections += 1
    start = time.time()

    try:
        print(f"[Worker {worker_id}] Processing request {request.id}")

        context = retrieve_context(request.query)
        result = run_llm(request.query, context)

        if not is_alive:
            raise HTTPException(
                status_code=503,
                detail=f"Worker {worker_id} failed during execution"
            )

        end = time.time()
        latency = end - start
        completed_tasks += 1

        return {
            "id": request.id,
            "worker_id": worker_id,
            "query": request.query,
            "result": result,
            "start_time": start,
            "end_time": end,
            "latency": latency,
            "status": "success",
            "error": ""
        }

    except Exception as e:
        failed_tasks += 1
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        active_connections -= 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)

    args = parser.parse_args()

    worker_id = args.worker_id

    print(f"[Worker {worker_id}] Starting REST API on port {args.port}")

    uvicorn.run(app, host="0.0.0.0", port=args.port)
    