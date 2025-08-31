import chainlit as cl
from dotenv import load_dotenv

from langchain.output_parsers import PydanticOutputParser
from memory import load_memory, save_interaction
from student_chain import build_student_chain
from evaluator_chain import build_evaluator_chain
from qa_generator import generate_initial_qa
from models import StudentResponse, TeacherResponse, EvaluatorResponse
# from student_chain import StudentResponse

load_dotenv()

# simulator to simulate student and teacher bot
# evaluator which evaluates the responses --> first
# genetic/evolutionary algo to generate syntethic data to be used in simulator


@cl.on_chat_start
async def start():
    """Initialize memory, chain, and seed questions when chat starts."""
    student_memory = load_memory("student")
    student_chain, vs = build_student_chain()
    evaluator_chain, vs = build_evaluator_chain()

    # Generate initial Q&A pool
    qa_pool = generate_initial_qa(vs, n=5)
    cl.user_session.set("qa_pool", qa_pool)
    cl.user_session.set("qa_index", 0)
    cl.user_session.set("student_chain", student_chain)
    cl.user_session.set("evaluator_chain", evaluator_chain)
    cl.user_session.set("student_memory", student_memory)

    # Student initiates conversation
    if qa_pool:
        first_q = qa_pool[0].q
        await cl.Message(content=f"👩‍🎓 Student: {first_q}").send()
    else:
        await cl.Message(content="👩‍🎓 Student: I don’t have any questions yet.").send()


@cl.on_message
async def main(message: cl.Message):
    """Handle teacher responses, student follow-up, and evaluator assessment."""

    # Load session objects
    student_chain = cl.user_session.get("student_chain")
    evaluator_chain = cl.user_session.get("evaluator_chain")
    student_memory = load_memory("student")
    qa_pool = cl.user_session.get("qa_pool", [])
    qa_index = cl.user_session.get("qa_index", 0)

    # Teacher provides explanation
    teacher_explanation = message.content
    teacher_model = TeacherResponse(message=teacher_explanation)
    interaction_id = save_interaction("teacher", teacher_model, interaction_id=None)

    # Expected answer from QA pool
    expected_explanation= qa_pool[qa_index].a if qa_index < len(qa_pool) else ""

    # Student generates response
    student_llm_response = student_chain.invoke({
        "student_memory": student_memory,
        "teacher_explanation": teacher_explanation
    })

    # Parse StudentResponse
    if isinstance(student_llm_response['text'], StudentResponse):
        student_model = student_llm_response['text']
    else:
        student_model = StudentResponse.parse_raw(student_llm_response['text'])
    save_interaction("student", student_model, interaction_id)

    # Send follow-up message
    if student_model.message:
        await cl.Message(content=f"👩‍🎓 Student: {student_model.message}").send()
    else:
        await cl.Message(content="👩‍🎓 Student: I think I understood this topic.").send()
        # Move to next QA pool question
        qa_index += 1
        cl.user_session.set("qa_index", qa_index)
        if qa_index < len(qa_pool):
            next_q = qa_pool[qa_index].q
            await cl.Message(content=f"👩‍🎓 Student: {next_q}").send()

# -------------------------------
# Evaluator assessment
# -------------------------------
    student_questions = []

# Original QA pool question (if available)
    if qa_index > 0:
        student_questions.append(qa_pool[qa_index-1].q)

# Actual student follow-up question (if any)
    if student_model.message:
        student_questions.append(student_model.message)

    evaluation_input = {
        "expected_explanation": expected_explanation,
        "teacher_explanation": teacher_explanation,
        "student_questions": student_questions,
        "student_response": student_model.dict()
    }

    evaluator_llm_response = evaluator_chain.invoke(evaluation_input)
    evaluator_model = EvaluatorResponse.parse_obj(evaluator_llm_response['text'])
    save_interaction("evaluator", evaluator_model, interaction_id)
