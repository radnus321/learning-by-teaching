from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from config import VS_DIR
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class RatingEnum(str, Enum):
    UNDERSTOOD = "understood"
    NEEDS_WORK = "needs work"
    CONFUSED = "confused"


class StudentResponse(BaseModel):
    question: Optional[str] = Field(
        None, description="A single follow-up question if the student is not fully satisfied"
    )
    missing_points: List[str] = Field(
        default_factory=list,
        description="Key gaps or missing explanations the student noticed"
    )
    rating: RatingEnum = Field(
        ..., description="Student's self-assessment of understanding"
    )


def build_student_chain():
    """Return (LLM chain, vectorstore)."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vs = Chroma(persist_directory=str(VS_DIR), embedding_function=embeddings)
    parser = PydanticOutputParser(pydantic_object=StudentResponse)

    student_template = """
    You are a curious student. Based on the teacher’s explanation and the context,
    decide if you understood the concept or if you need clarification.

    - If you understood fully → set rating = "understood" and do not ask a follow-up.
    - If partially clear → set rating = "needs Work" and ask exactly ONE concise follow-up question.
    - If confused → set rating = "confused" and ask exactly ONE clarifying question.

    Respond ONLY in JSON valid for the StudentResponse model:
    {{
      "question": "Your one follow-up question, or null if none",
      "missing_points": ["Point 1", "Point 2"],
      "rating": "understood" | "needs work" | "confused"
    }}

    - The rating has a type enum and supports only 3 values: ["understood", "needs work", "confused"]

    Context:
    {context}

    Teacher explanation:
    {teacher_explanation}
    """

    prompt = ChatPromptTemplate.from_template(
        student_template
    ).partial(format_instructions=parser.get_format_instructions())

    chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)

    return chain, vs
