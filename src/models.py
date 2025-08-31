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
class EvaluatorResponse(BaseModel):
    rating: str  # "excellent" | "good" | "partial" | "needs work" | "incorrect"
    missing_points: Optional[List[str]] = []
    incorrect_points: Optional[List[str]] = []
    feedback: Optional[str] = None
    referenced_points: Optional[List[str]] = []


# -------------------------------
# Scorer
# -------------------------------
class ScorerResponse(BaseModel):
    value: int  # Numerical score (0-10, or scale we define)
    feedback: str
