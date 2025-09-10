from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from models import ScorerResponse


scorer_prompt = """
You are an automated scorer for a teaching-learning interaction.
You are given the following details of a single interaction:

- teacher_explanation: What the teacher explained.
- student_question: The initial question asked by the student.
- student_followup_question: Any follow-up question from the student.
- student_response: The response of the student (message, rating, reflection, missing_points).
- evaluator_comments: Qualitative feedback from the evaluator.

Your task is to produce a **quantitative evaluation** of the interaction as a JSON object
matching the following Pydantic model `ScorerResponse`:

{{
  "overall_score": "float between 0.0 and 1.0",
  "teacher_clarity": "float between 0.0 and 1.0",
  "teacher_completeness": "float between 0.0 and 1.0",
  "student_understanding": "float between 0.0 and 1.0",
  "student_engagement": "float between 0.0 and 1.0",
  "comments": ["string"]
}}

Rules:

1. All scores must be **between 0.0 and 1.0** inclusive.
2. Provide all scores; do not leave any field blank.
3. Comments should summarize any important qualitative insights.
4. Respond **ONLY** in valid JSON corresponding to the `ScorerResponse` model.

Interaction details:

teacher_explanation: {teacher_explanation}
student_question: {student_question}
student_followup_question: {student_followup_question}
student_response: {student_response}
evaluator_comments: {evaluator_comments}
"""


def build_scorer_chain(llm):
    """Build the LLM chain for scoring interactions."""
    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4)
    parser = PydanticOutputParser(pydantic_object=ScorerResponse)

    prompt = ChatPromptTemplate.from_template(scorer_prompt).partial(
        format_instructions=parser.get_format_instructions()
    )

    chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
    return chain
