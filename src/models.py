from typing import List, Optional, Literal
from pydantic import BaseModel


# -------------------------------
# Teacher
# -------------------------------
class TeacherResponse(BaseModel):
    message: str  # Explanation of the concept


# -------------------------------
# Student
# -------------------------------
class StudentResponse(BaseModel):
    message: Optional[str]  # Follow-up question, can be null if fully understood
    rating: Literal["understood", "needs work", "confused"]
    reflection: str  # Student’s meta-understanding: e.g. "I didn’t understand sorting properly."
    missing_points: List[str] = []  # Gaps in knowledge


# -------------------------------
# Evaluator
# -------------------------------
class EvaluationResponse(BaseModel):
    correctness: Literal["correct", "partial", "incorrect"]
    missing_points: List[str]
    feedback: str


# -------------------------------
# Scorer
# -------------------------------
class ScorerResponse(BaseModel):
    value: int  # Numerical score (0-10, or scale we define)
    feedback: str
