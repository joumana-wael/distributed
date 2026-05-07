from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from Master.master_client import MasterClient
from lb.global_load_balancer import GlobalLoadBalancer


app = FastAPI(title="Global Load Balancer API")


class UserRequest(BaseModel):
    id: int
    query: str


master1 = MasterClient(
    master_id=1,
    base_url="http://127.0.0.1:9001"
)

master2 = MasterClient(
    master_id=2,
    base_url="http://127.0.0.1:9002"
)

global_lb = GlobalLoadBalancer(
    masters=[master1, master2],
    strategy="global_round_robin"
)


@app.get("/health")
def health():
    return {
        "service": "Global Load Balancer",
        "status": "running",
        "strategy": global_lb.strategy,
        "masters": [
            {
                "master_id": master1.master_id,
                "available": master1.is_available()
            },
            {
                "master_id": master2.master_id,
                "available": master2.is_available()
            }
        ]
    }


@app.post("/ask")
def ask(request: UserRequest):
    try:
        response = global_lb.dispatch(request)
        return response

    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)