import uuid
import json
from pathlib import Path
from pydantic import BaseModel

MEMORY_DIR = Path("memory")
MEMORY_DIR.mkdir(exist_ok=True)
FILES = {
    "teacher": MEMORY_DIR / "teacher.json",
    "student": MEMORY_DIR / "student.json",
    "evaluator": MEMORY_DIR / "evaluator.json",
    "scorer": MEMORY_DIR / "scorer.json",
}


def load_memory(agent: str):
    file = FILES[agent]
    if not file.exists():
        return []
    return json.loads(file.read_text(encoding="utf-8"))


def save_interaction(agent: str, model: BaseModel, interaction_id: str = None):
    """Save validated agent response to its memory file."""
    interaction_id = interaction_id or str(uuid.uuid4())
    file = FILES[agent]

    # load existing
    if file.exists():
        memory = json.loads(file.read_text(encoding="utf-8"))
    else:
        memory = []

    # append validated dict
    memory.append({
        "interaction_id": interaction_id,
        agent: model.dict()
    })

    file.write_text(json.dumps(memory, indent=2,
                    ensure_ascii=False), encoding="utf-8")
    return interaction_id
