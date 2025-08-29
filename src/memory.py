import json
from config import MEMORY_FILE
from student_chain import StudentResponse


def load_memory():
    """Load memory from JSON file."""
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def save_interaction(memory, teacher_explanation: str, student_response):
    """Save teacher explanation and structured student response."""
    entry = {
        "teacher": teacher_explanation,
        "student": student_response.dict()
    }

    print(entry)

    memory.append(entry)

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

    return entry
