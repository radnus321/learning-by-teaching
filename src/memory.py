import json
from config import MEMORY_FILE


def load_memory():
    """Load memory from JSON file."""
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def save_interaction(memory, teacher_explanation, student_response):
    """Parse & store structured student response."""
    try:
        cleaned = student_response.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(
                line for line in cleaned.splitlines()
                if not line.strip().startswith("```")
            )
        parsed = json.loads(cleaned)
        questions = parsed.get("questions", [])
        missing_points = parsed.get("missing_points", [])
        rating = parsed.get("rating", "Unknown")
    except json.JSONDecodeError:
        questions, missing_points, rating = [], [], "Unknown"

    memory.append({
        "teacher": teacher_explanation,
        "student": {
            "questions": questions,
            "missing_points": missing_points,
            "rating": rating
        }
    })

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)
