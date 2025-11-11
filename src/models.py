import os
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, confloat, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")


def get_llm(model_name: str):
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
        model=f"{model_name}"
    )

# -------------------------------
# Teacher
# -------------------------------
class TeacherResponse(BaseModel):
    message: str = "Here’s an explanation of the concept."  # Default placeholder


# -------------------------------
# Student
# -------------------------------
class StudentResponse(BaseModel):
    message: Optional[str] = None  # None means no follow-up question
    rating: Literal["understood", "needs work", "confused"] = "understood"
    reflection: str = "I understood the concept well."
    missing_points: List[str] = Field(default_factory=list)


# -------------------------------
# Evaluator
# -------------------------------
class EvaluatorResponse(BaseModel):
    rating: str = "good"  # default rating
    missing_points: Optional[List[str]] = Field(default_factory=list)
    incorrect_points: Optional[List[str]] = Field(default_factory=list)
    feedback: Optional[str] = "Good explanation overall."
    referenced_points: Optional[List[str]] = Field(default_factory=list)


# -------------------------------
# Scorer
# -------------------------------
class ScorerResponse(BaseModel):
    overall_score: confloat(ge=0.0, le=1.0) = 0
    teacher_clarity: confloat(ge=0.0, le=1.0) = 0
    teacher_completeness: confloat(ge=0.0, le=1.0) = 0
    student_understanding: confloat(ge=0.0, le=1.0) = 0
    student_engagement: confloat(ge=0.0, le=1.0) = 0
    comments: List[str] = Field(default_factory=lambda: ["Good session. Minor gaps noted."])


# -------------------------------
# Final Score
# -------------------------------
class FinalScorerResponse(BaseModel):
    overall_score: confloat(ge=0.0, le=1.0) = 0
    teacher_clarity: confloat(ge=0.0, le=1.0) = 0
    teacher_completeness: confloat(ge=0.0, le=1.0) = 0
    student_understanding: confloat(ge=0.0, le=1.0) = 0
    student_engagement: confloat(ge=0.0, le=1.0) = 0
    comments: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "teacher": [""],
            "student": [""],
            "evaluator": [""]
        }
    )
