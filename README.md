# Distributed RAG + LLM GPU Worker System

This project implements a distributed request-processing architecture for RAG + LLM inference using:

- Global Load Balancer
- Master Nodes
- GPU Worker Nodes
- REST API communication
- RAG retrieval
- Real LLM inference using Hugging Face Transformers
- Optional simulated inference mode for large-scale load testing
- Fault tolerance for worker and master node failures

---

## 1. System Architecture

```text
            Client / Postman / Load Test
                        |
                        v
           Global Load Balancer API :7000
                        |
            +----------------------+
            |                      |
            v                      v
Master Node 1 :9001          Master Node 2 :9002
        |                              |
        v                              v
Workers 0,1                        Workers 2,3
:8001,:8002                        :8003,:8004
```

Each worker can run either:

1. **Real LLM mode** using Hugging Face + PyTorch CUDA/CPU.
2. **Simulated inference mode** for large-scale load testing.

---

## 2. Recommended Environment

Recommended versions:

```text
Python: 3.10 or 3.11 recommended
pip: 23.0 or newer
OS: Windows / Linux / Thunder Compute GPU instance
GPU: NVIDIA GPU with CUDA support for real GPU inference
```

Check your Python version:

```bash
python --version
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

---

## 3. Install Dependencies

Install the required packages:

```bash
python -m pip install fastapi "uvicorn[standard]"
python -m pip install requests
python -m pip install transformers
python -m pip install torch
python -m pip install pynvml
python -m pip install numpy
```

Optional, if using environment files:

```bash
python -m pip install python-dotenv
```

### CUDA / GPU Check

Run:

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
```

Expected on a GPU instance:

```text
CUDA Available: True
GPU: NVIDIA ...
```

You can also check GPU status using:

```bash
nvidia-smi
```

---

## 4. Project Startup - Local / Forwarded-Port Setup

Open a separate terminal for each service.

### 4.1 Start Worker Nodes

```bash
python -m Workers.worker_api --worker-id 0 --port 8001
python -m Workers.worker_api --worker-id 1 --port 8002
python -m Workers.worker_api --worker-id 2 --port 8003
python -m Workers.worker_api --worker-id 3 --port 8004
```

Optional extra worker:

```bash
python -m Workers.worker_api --worker-id 4 --port 8005
```

### 4.2 Start Master Nodes

Master 1 controls Workers 0 and 1:

```bash
python -m Master.master_api --master-id 1 --port 9001 --worker-start 0 --worker-count 2 --worker-port-start 8001
```

Master 2 controls Workers 2 and 3:

```bash
python -m Master.master_api --master-id 2 --port 9002 --worker-start 2 --worker-count 2 --worker-port-start 8003
```

### 4.3 Start the Global Load Balancer

```bash
python -m lb.global_lb_api
```

The Global Load Balancer should run on:

```text
http://127.0.0.1:7000
```

---

## 5. Startup on Multiple GPU Worker Instances

If workers are deployed on separate Thunder Compute GPU instances, each worker can run on port `800x` on its own machine.

On each worker instance:

```bash
python -m Workers.worker_api --worker-id 0 --port 8001
```
using worker ids 0, 1, 2, 3 and ports 8001, 8002, 8003, 8004

To find the worker machine IP:

```bash
hostname -I
```

Example worker URLs:

```text
Worker 0 -> http://WORKER0_IP:8001
Worker 1 -> http://WORKER1_IP:8001
Worker 2 -> http://WORKER2_IP:8001
Worker 3 -> http://WORKER3_IP:8001
```

---

## 6. Real LLM Mode vs Simulated Mode

### 6.1 Real LLM Mode

Use this mode to prove:

- RAG retrieval works
- LLM inference works
- CUDA/GPU execution works
- Real REST request flow works

### 6.2 Simulated Inference Mode

Use this mode for large-scale load testing such as 100 to 1000 users.

#### PowerShell

```powershell
$env:SIMULATE_INFERENCE="1"
$env:SIM_MIN_DELAY="0.05"
$env:SIM_MAX_DELAY="0.20"
python -m Workers.worker_api --worker-id 0 --port 8001
```

#### Linux / Thunder

```bash
export SIMULATE_INFERENCE=1
export SIM_MIN_DELAY=0.05
export SIM_MAX_DELAY=0.20
python -m Workers.worker_api --worker-id 0 --port 8001
```

In simulated mode, the worker returns a simulated LLM response after a controlled delay.

---

## 7. Health Check Commands

### Global Load Balancer

```powershell
Invoke-RestMethod -Method GET -Uri http://127.0.0.1:7000/health
```

### Worker 0

```powershell
Invoke-RestMethod -Method GET -Uri http://127.0.0.1:8001/health
```

### Worker 1

```powershell
Invoke-RestMethod -Method GET -Uri http://127.0.0.1:8002/health
```

### Worker 2

```powershell
Invoke-RestMethod -Method GET -Uri http://127.0.0.1:8003/health
```

### Worker 3

```powershell
Invoke-RestMethod -Method GET -Uri http://127.0.0.1:8004/health
```

### Master 1

```powershell
Invoke-RestMethod -Method GET -Uri http://127.0.0.1:9001/health
```

### Master 2

```powershell
Invoke-RestMethod -Method GET -Uri http://127.0.0.1:9002/health
```

---

## 8. Send a Real Request

Use this to test the full path:

```text
Client -> Global LB -> Master -> Worker -> RAG -> LLM -> Response
```

PowerShell:

```powershell
Invoke-RestMethod -Method POST -Uri http://127.0.0.1:7000/ask -ContentType "application/json" -Body '{"id":1,"query":"Explain how worker fault tolerance works in this system."}'
```

Expected fields:

```text
id
worker_id
master_id
result
latency
gpu_utilization
inference_mode
status
```

Example expected result:

```text
status          : success
inference_mode  : real_llm or simulated
master_id       : 1 or 2
worker_id       : 0 / 1 / 2 / 3
```
Or use Postman throught the following steps:

```text
1 - Open Postman extension
2 - New HTTP request
3 - Set method = POST and URL = "http://127.0.0.1:7000/ask"
4 - Select Raw and type = "JSON"
5 - Add the request body using the id and query headers
```
---

## 9. Fault Tolerance Testing

### 9.1 Worker Failure Test

Fail Worker 0:

```powershell
Invoke-RestMethod -Method POST -Uri http://127.0.0.1:8001/fail
```

Check Worker 0:

```powershell
Invoke-RestMethod -Method GET -Uri http://127.0.0.1:8001/health
```

Recover Worker 0:

```powershell
Invoke-RestMethod -Method POST -Uri http://127.0.0.1:8001/recover
```

### 9.2 Master Failure Test

Fail Master 1:

```powershell
Invoke-RestMethod -Method POST -Uri http://127.0.0.1:9001/fail
```

Check Master 1:

```powershell
Invoke-RestMethod -Method GET -Uri http://127.0.0.1:9001/health
```

Recover Master 1:

```powershell
Invoke-RestMethod -Method POST -Uri http://127.0.0.1:9001/recover
```

### 9.3 Hard Shutdown Test

To simulate a hard shutdown, stop the worker or master process directly using:

```text
Ctrl + C
```

Then run a request or load test again. The system should skip the failed node and continue using healthy nodes.

---

## 10. Load Testing

### 10.1 REST Load Testing Through Real APIs

Use this when workers are running as REST APIs.

Example:

```bash
python client/rest_load_test.py --levels 100 --query "Simulated large-scale inference request"
```

Run 100 to 1000 users:

```bash
python client/rest_load_test.py --levels 100 250 500 1000 --query "Simulated large-scale inference request"
```

This tests the real REST architecture:

```text
Load Test -> Global LB -> Masters -> Workers
```

Use simulated worker mode for 100 to 1000 users.

### 10.2 In-Memory Simulated Load Test

If using the local simulated script:

```bash
python client/simulated_load_test.py --levels 100 250 500 1000
```

Compare strategies:

```bash
python client/simulated_load_test.py --levels 100 250 500 1000 --strategy round_robin
python client/simulated_load_test.py --levels 100 250 500 1000 --strategy least_connections
python client/simulated_load_test.py --levels 100 250 500 1000 --strategy load_aware
```

---

## 11. GPU Monitoring

Run this on the GPU worker instance:

```bash
nvidia-smi
```

Continuous monitoring:

```bash
nvidia-smi -l 1
```

Show Python GPU memory usage:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv -l 1
```

Save GPU metrics to CSV:

```bash
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 1 > gpu_metrics.csv
```

Stop with:

```text
Ctrl + C
```

---

## 12. Expected Successful Results

### Real LLM/GPU Test

Expected:

```text
status          : success
inference_mode  : real_llm
worker_id       : 0 / 1 / 2 / 3
master_id       : 1 / 2
```

### Simulated 1000-User Test

Expected example:

```text
Total requests       : 1000
Successful requests  : 1000
Failed requests      : 0
Requests per master:
Master 1: 500
Master 2: 500

Requests per worker:
Worker 0: 250
Worker 1: 250
Worker 2: 250
Worker 3: 250
```

---

## 13. Recommended Final Testing Order

1. Start all workers.
2. Start Master 1 and Master 2.
3. Start Global Load Balancer.
4. Check `/health` for all components.
5. Send one real request to `/ask`.
6. Validate `inference_mode`.
7. Run real LLM/GPU test with small load.
8. Switch workers to simulated mode.
9. Run 100, 250, 500, and 1000 user load tests.
10. Run worker failure test.
11. Run master failure test.
12. Run combined failure test.
