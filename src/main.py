import chainlit as cl
from dotenv import load_dotenv
from memory import load_memory, save_interaction
from student_chain import build_student_chain
from evaluator_chain import build_evaluator_chain
from scorer_chain import build_scorer_chain
from qa_generator import generate_initial_qa
from models import StudentResponse, TeacherResponse, EvaluatorResponse, ScorerResponse

load_dotenv()


@cl.on_chat_start
async def start():
    """Initialize memory, chains, and QA pool."""
    student_memory = load_memory("student")
    teacher_memory = load_memory("teacher")
    evaluator_memory = load_memory("evaluator")
    scorer_memory = load_memory("scorer")

    student_chain, vs = build_student_chain()
    evaluator_chain = build_evaluator_chain()
    scorer_chain = build_scorer_chain()

    # Generate initial Q&A pool
    qa_pool = generate_initial_qa(vs, n=5)

    # Store in session
    cl.user_session.set("student_chain", student_chain)
    cl.user_session.set("evaluator_chain", evaluator_chain)
    cl.user_session.set("scorer_chain", scorer_chain)
    cl.user_session.set("student_memory", student_memory)
    cl.user_session.set("teacher_memory", teacher_memory)
    cl.user_session.set("evaluator_memory", evaluator_memory)
    cl.user_session.set("scorer_memory", scorer_memory)
    cl.user_session.set("qa_pool", qa_pool)
    cl.user_session.set("qa_index", 0)

    # Student initiates conversation
    if qa_pool:
        first_q = qa_pool[0].q
        await cl.Message(content=f"👩‍🎓 Student: {first_q}").send()
    else:
        await cl.Message(content="👩‍🎓 Student: I don’t have any questions yet.").send()


@cl.on_message
async def main(message: cl.Message):
    """Handle teacher input, student response, evaluation, and scoring."""
    student_chain = cl.user_session.get("student_chain")
    evaluator_chain = cl.user_session.get("evaluator_chain")
    scorer_chain = cl.user_session.get("scorer_chain")
    student_memory = cl.user_session.get("student_memory")
    teacher_memory = cl.user_session.get("teacher_memory")
    evaluator_memory = cl.user_session.get("evaluator_memory")
    scorer_memory = cl.user_session.get("scorer_memory")
    qa_pool = cl.user_session.get("qa_pool", [])
    qa_index = cl.user_session.get("qa_index", 0)

    # 1️⃣ Teacher provides explanation
    teacher_explanation = message.content
    teacher_model = TeacherResponse(message=teacher_explanation)
    interaction_id = save_interaction("teacher", teacher_model)

    # Expected answer from QA pool
    expected_answer = qa_pool[qa_index].a if qa_index < len(qa_pool) else ""

    # 2️⃣ Student generates response
    student_llm_response = student_chain.invoke({
        "student_memory": student_memory,
        "teacher_explanation": teacher_explanation
    })

    # Parse student response
    if isinstance(student_llm_response['text'], StudentResponse):
        student_model = student_llm_response['text']
    else:
        student_model = StudentResponse.parse_raw(student_llm_response['text'])
    save_interaction("student", student_model, interaction_id)

    # 3️⃣ Evaluator assesses interaction
    evaluator_llm_response = evaluator_chain.invoke({
        "expected_explanation": expected_answer,
        "teacher_explanation": teacher_explanation,
        "student_question": qa_pool[qa_index].q,
        "student_followup_question": student_model.message,
        "student_response": student_model.json()
    })

    # Parse Evaluator response
    if isinstance(evaluator_llm_response['text'], EvaluatorResponse):
        evaluator_model = evaluator_llm_response['text']
    else:
        evaluator_model = EvaluatorResponse.parse_raw(evaluator_llm_response['text'])
    save_interaction("evaluator", evaluator_model, interaction_id)

    # 4️⃣ Scorer computes quantitative metrics
    scorer_llm_response = scorer_chain.invoke({
        "teacher_explanation": teacher_explanation,
        "student_question": qa_pool[qa_index].q,
        "student_followup_question": student_model.message,
        "student_response": student_model.json(),
        "evaluator_comments": evaluator_model.json()
    })
    if isinstance(scorer_llm_response['text'], ScorerResponse):
        scorer_model = scorer_llm_response['text']
    else:
        scorer_model = ScorerResponse.parse_raw(scorer_llm_response)
    save_interaction("scorer", scorer_model, interaction_id)

    # 5️⃣ Send follow-up if student asks
    if student_model.message:
        await cl.Message(content=f"👩‍🎓 Student: {student_model.message}").send()
    else:
        await cl.Message(content="👩‍🎓 Student: I think I understood this topic.").send()
        # Move to next QA
        qa_index += 1
        cl.user_session.set("qa_index", qa_index)
        if qa_index < len(qa_pool):
            next_q = qa_pool[qa_index].q
            await cl.Message(content=f"👩‍🎓 Student: {next_q}").send()
