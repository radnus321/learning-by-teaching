import uuid
import json
from pathlib import Path
from pydantic import BaseModel
import datetime
from db import (
    interaction_collection,
    teacher_collection,
    student_collection,
    evaluator_collection,
    scorer_collection,
)

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


def create_interaction(user_id: str) -> str:
    """Create a new interaction entry for a user."""
    interaction_id = str(uuid.uuid4())
    interaction_collection.insert_one({
        "_id": interaction_id,
        "user_id": user_id,
        "timestamp": datetime.utcnow()
    })
    return interaction_id


def save_teacher(interaction_id: str, model: BaseModel):
    teacher_collection.insert_one({
        "_id": interaction_id,
        **model.dict(),
        "timestamp": datetime.utcnow()
    })


def save_student(interaction_id: str, model: BaseModel):
    student_collection.insert_one({
        "_id": interaction_id,
        **model.dict(),
        "timestamp": datetime.utcnow()
    })


def save_evaluator(interaction_id: str, model: BaseModel):
    evaluator_collection.insert_one({
        "_id": interaction_id,
        **model.dict(),
        "timestamp": datetime.utcnow()
    })


def save_scorer(interaction_id: str, model: BaseModel):
    scorer_collection.insert_one({
        "_id": interaction_id,
        **model.dict(),
        "timestamp": datetime.utcnow()
    })
