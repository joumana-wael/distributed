import threading
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-small"

_tokenizer = None
_model = None

_load_lock = threading.Lock()
_inference_lock = threading.Lock()


def load_model():
    global _tokenizer, _model

    with _load_lock:
        if _tokenizer is None or _model is None:
            print("[LLM] Loading real model...")

            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

            _model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
                device_map=None
            )

            _model.to("cpu")
            _model.eval()

            print("[LLM] Model loaded successfully.")

    return _tokenizer, _model


def preload_model():
    load_model()


def run_llm(query, context):
    tokenizer, model = load_model()

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

    with _inference_lock:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {key: value.to("cpu") for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer