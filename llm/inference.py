import os
import random
import re
import threading
import time

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-small"

# Simulation mode for large-scale load testing
SIMULATE_INFERENCE = os.getenv("SIMULATE_INFERENCE", "0") == "1"
SIM_MIN_DELAY = float(os.getenv("SIM_MIN_DELAY", "0.05"))
SIM_MAX_DELAY = float(os.getenv("SIM_MAX_DELAY", "0.20"))

# Automatically use GPU if CUDA is available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

_tokenizer = None
_model = None

_load_lock = threading.Lock()
_inference_lock = threading.Lock()


def load_model():
    global _tokenizer, _model

    with _load_lock:
        if _tokenizer is None or _model is None:
            print("[LLM] Loading real model...")
            print(f"[LLM] Simulation mode: {SIMULATE_INFERENCE}")
            print(f"[LLM] CUDA available: {torch.cuda.is_available()}")
            print(f"[LLM] Using device: {DEVICE}")

            if DEVICE == "cuda":
                print(f"[LLM] GPU name: {torch.cuda.get_device_name(0)}")

            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

            _model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME,
                dtype=MODEL_DTYPE,
                low_cpu_mem_usage=False,
                device_map=None
            )

            _model.to(DEVICE)
            _model.eval()

            print(f"[LLM] Model parameter device: {next(_model.parameters()).device}")
            print("[LLM] Model loaded successfully.")

    return _tokenizer, _model


def preload_model():
    if SIMULATE_INFERENCE:
        print("[LLM] Simulation mode enabled. Real model preload skipped.")
        return

    load_model()


def extract_direct_answer(context):
    """
    If the RAG context contains a direct answer line like:
    Answer: 2+2 = 4.
    return that exact answer for simple factual/math questions.
    """

    for line in context.splitlines():
        line = line.strip()

        if line.lower().startswith("answer:"):
            answer = line.split(":", 1)[1].strip()

            if answer:
                return answer

    return None


def is_math_or_exact_fact_query(query):
    """
    Only bypass the LLM for simple math/exact fact queries.
    For explanation questions, the LLM should still generate the answer.
    """

    query_lower = query.lower()

    if re.search(r"\d+\s*[\+\-\*/]\s*\d+", query_lower):
        return True

    return False


def is_weak_answer(answer):
    cleaned = answer.strip().lower()

    weak_answers = {
        "",
        "a",
        "a).",
        "b",
        "c",
        "d",
        "yes",
        "no",
        "(i)",
        "(ii)",
        "(iii)",
        "(iv)",
        "i",
        "ii",
        "iii",
        "iv"
    }

    if cleaned in weak_answers:
        return True

    if len(cleaned.split()) < 4:
        return True

    if re.fullmatch(r"[\(\)\[\]\{\}\.\,\:\;\-\w]{1,8}", cleaned):
        return True

    return False


def looks_repetitive(answer):
    """
    Detects repeated-generation behavior from small models.
    Example:
    '2+2 = 4 - 2 = 4 - 2 = 4'
    """

    cleaned = answer.strip().lower()

    parts = re.split(r"\s*[-|;]\s*", cleaned)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) >= 3:
        unique_parts = set(parts)

        if len(unique_parts) <= 2:
            return True

    tokens = cleaned.split()

    if len(tokens) >= 8:
        most_common_count = max(tokens.count(token) for token in set(tokens))

        if most_common_count >= 4:
            return True

    return False


def build_fallback_answer(context):
    context = context.strip()

    if not context:
        return "The knowledge base does not contain enough information to answer this."

    return f"Based on the retrieved context: {context}"


def run_llm(query, context):
    """
    Runs LLM inference.

    Modes:
    1. Real mode:
       - Uses Hugging Face model.
       - Uses CUDA if available.
       - Falls back to CPU if CUDA is unavailable.

    2. Simulation mode:
       - Enabled with SIMULATE_INFERENCE=1.
       - Does not load the real model.
       - Sleeps for a controlled delay.
       - Used for large-scale load testing.
    """

    if SIMULATE_INFERENCE:
        delay = random.uniform(SIM_MIN_DELAY, SIM_MAX_DELAY)
        time.sleep(delay)

        return {
            "answer": (
                "SIMULATED_GPU_WORKER_RESPONSE: "
                "This response simulates LLM inference delay for large-scale load testing."
            ),
            "gpu_utilization": get_gpu_utilization(),
            "mode": "simulated"
        }

    direct_answer = extract_direct_answer(context)

    if direct_answer and is_math_or_exact_fact_query(query):
        return {
            "answer": direct_answer
        }

    tokenizer, model = load_model()

    prompt = f"""
You are a technical assistant for a distributed computing project.

Use ONLY the context below to answer the user's question.
Write a complete and clear answer in 3 to 5 sentences.
Do not answer with a short phrase.
Do not answer with multiple-choice letters.
Explain using the retrieved context.

Context:
{context}

Question:
{query}

Complete answer:
"""

    with _inference_lock:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                min_new_tokens=20,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5
            )

        answer = tokenizer.decode(
            outputs[0].detach().cpu(),
            skip_special_tokens=True
        ).strip()

    if is_weak_answer(answer) or looks_repetitive(answer):
        answer = build_fallback_answer(context)


    return {
        "answer": answer
    }