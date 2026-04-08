import json
import requests
from config import OLLAMA_BASE_URL, OLLAMA_MODEL


def evaluate_answer(question_text: str, correct_answer: str, user_answer: str) -> dict:
    """Evaluate a user's answer against the correct answer using Ollama.

    Returns a dict with keys: is_correct, mistake_type, explanation.
    """
    prompt = (
        f"You are evaluating an answer. The question is: {question_text}. "
        f"The correct answer is: {correct_answer}. The user answered: {user_answer}. "
        f"Determine if correct. If wrong, classify the mistake as one of: "
        f"conceptual, misinterpretation, calculation, partial understanding. "
        f"Provide a brief explanation. Return JSON with keys: "
        f"is_correct (boolean), mistake_type (string, null if correct), explanation (string)."
    )

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "format": "json",
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        parsed = json.loads(result["response"])
        return {
            "is_correct": parsed.get("is_correct", False),
            "mistake_type": parsed.get("mistake_type"),
            "explanation": parsed.get("explanation", ""),
        }
    except Exception:
        # Fallback: simple string-based evaluation
        is_correct = user_answer.strip().lower() == correct_answer.strip().lower()
        return {
            "is_correct": is_correct,
            "mistake_type": None if is_correct else "partial understanding",
            "explanation": (
                "Correct." if is_correct
                else "Ollama unavailable. Answers do not match exactly; unable to classify mistake type."
            ),
        }
