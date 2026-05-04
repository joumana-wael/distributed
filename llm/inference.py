import re
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
                dtype=torch.float32,
                low_cpu_mem_usage=False,
                device_map=None
            )

            _model.to("cpu")
            _model.eval()

            print("[LLM] Model loaded successfully.")

    return _tokenizer, _model


def preload_model():
    load_model()


def extract_direct_answer(context):
    """
    If the retrieved RAG context contains a direct answer line like:
    Answer: 2+2 = 4.
    return that answer directly instead of asking the small LLM to rewrite it.
    """
    for line in context.splitlines():
        line = line.strip()

        if line.lower().startswith("answer:"):
            answer = line.split(":", 1)[1].strip()

            if answer:
                return answer

    return None


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

    # Catches weak outputs like "(4).", "4.", "(a)", "option c", etc.
    if len(cleaned.split()) < 4:
        return True

    # Catches answers that are mostly symbols / option-like fragments.
    if re.fullmatch(r"[\(\)\[\]\{\}\.\,\:\;\-\w]{1,8}", cleaned):
        return True

    return False


def looks_repetitive(answer):
    """
    Detects simple repeated-generation behavior from small models.
    Example:
    '2+2 = 4 - 2 = 4 - 2 = 4'
    """
    cleaned = answer.strip().lower()

    # Repeated phrase around separators.
    parts = re.split(r"\s*[-|;]\s*", cleaned)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) >= 3:
        unique_parts = set(parts)

        if len(unique_parts) <= 2:
            return True

    # Too many repeated short expressions.
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
    # If the RAG context contains an explicit answer, return it directly.
    # This prevents small-model hallucination/repetition for exact facts.
    direct_answer = extract_direct_answer(context)

    if direct_answer:
        return direct_answer

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

        inputs = {key: value.to("cpu") for key, value in inputs.items()}

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

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if is_weak_answer(answer) or looks_repetitive(answer):
        answer = build_fallback_answer(context)

    return answer