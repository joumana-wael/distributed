import re
import threading
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates
)


MODEL_NAME = "google/flan-t5-small"

# Automatically use GPU if CUDA is available, otherwise use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

_tokenizer = None
_model = None

_load_lock = threading.Lock()
_inference_lock = threading.Lock()

# GPU monitoring globals
_gpu_initialized = False
_gpu_handle = None


def initialize_gpu_monitor():
    global _gpu_initialized, _gpu_handle

    if DEVICE == "cuda" and not _gpu_initialized:
        nvmlInit()
        _gpu_handle = nvmlDeviceGetHandleByIndex(0)
        _gpu_initialized = True


def get_gpu_utilization():
    """
    Returns current GPU utilization percentage.
    Returns 0 if running on CPU.
    """

    if DEVICE != "cuda":
        return 0

    initialize_gpu_monitor()

    utilization = nvmlDeviceGetUtilizationRates(_gpu_handle)

    return utilization.gpu


def load_model():
    global _tokenizer, _model

    with _load_lock:
        if _tokenizer is None or _model is None:
            print("[LLM] Loading real model...")
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

            # This proves where the model is actually loaded
            print(f"[LLM] Model parameter device: {next(_model.parameters()).device}")
            print("[LLM] Model loaded successfully.")

    return _tokenizer, _model


def preload_model():
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

    # Catches weak outputs like "(4).", "4.", "(a)", etc.
    if len(cleaned.split()) < 4:
        return True

    # Catches answers that are mostly symbols or option-like fragments
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
    Runs real LLM inference.

    If CUDA is available, the model and inputs are moved to GPU.
    If CUDA is not available, it falls back to CPU.

    For direct math/exact facts, it can return the RAG answer directly.
    For explanation questions, it runs the model.
    """

    direct_answer = extract_direct_answer(context)

    if direct_answer and is_math_or_exact_fact_query(query):
        return {
            "answer": direct_answer,
            "gpu_utilization": get_gpu_utilization()
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

        # Move input tensors to the same device as the model
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

        # Move output back to CPU before decoding
        answer = tokenizer.decode(
            outputs[0].detach().cpu(),
            skip_special_tokens=True
        ).strip()

    if is_weak_answer(answer) or looks_repetitive(answer):
        answer = build_fallback_answer(context)

    gpu_util = get_gpu_utilization()

    return {
        "answer": answer,
        "gpu_utilization": gpu_util
    }