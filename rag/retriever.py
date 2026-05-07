import os
import re


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "knowledge_base.txt")


STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "to", "in", "on",
    "for", "and", "or", "with", "this", "that", "how", "does", "do",
    "it", "as", "by", "from", "at"
}


def tokenize(text):
    tokens = re.findall(r"\b\w+\b", text.lower())

    return set(
        token
        for token in tokens
        if token not in STOPWORDS
    )


def normalize_math(text):
    return text.lower().replace(" ", "")


def retrieve_context(query, top_k=3):
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        return "No knowledge base found."

    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as file:
        paragraphs = [
            p.strip()
            for p in file.read().split("\n\n")
            if p.strip()
        ]

    query_tokens = tokenize(query)

    normalized_query = normalize_math(query)

    scored = []

    for paragraph in paragraphs:
        paragraph_tokens = tokenize(paragraph)

        normalized_paragraph = normalize_math(paragraph)

        score = 0

        # -----------------------------
        # General keyword overlap
        # -----------------------------
        overlap = len(
            query_tokens.intersection(paragraph_tokens)
        )

        if paragraph_tokens:
            score += overlap / len(paragraph_tokens)

        # -----------------------------
        # Strong boost for Question: match
        # -----------------------------
        question_match = re.search(
            r"Question:\s*(.*)",
            paragraph,
            re.IGNORECASE
        )

        if question_match:
            question_text = question_match.group(1)

            question_tokens = tokenize(question_text)

            question_overlap = len(
                query_tokens.intersection(question_tokens)
            )

            score += question_overlap * 10

        # -----------------------------
        # Strong boost for exact math
        # -----------------------------
        math_expressions = re.findall(
            r"\d+\s*[\+\-\*/]\s*\d+",
            query
        )

        for expr in math_expressions:
            if normalize_math(expr) in normalized_paragraph:
                score += 100

        # -----------------------------
        # Strong boost for exact query
        # -----------------------------
        if normalized_query in normalized_paragraph:
            score += 50

        if score > 0:
            scored.append((score, paragraph))

    scored.sort(
        reverse=True,
        key=lambda item: item[0]
    )

    if not scored:
        return "No relevant context found in the knowledge base."

    return "\n".join(
        paragraph
        for score, paragraph in scored[:top_k]
    )