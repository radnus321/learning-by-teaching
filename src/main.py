import os
from datetime import datetime
import uuid
import chainlit as cl
from chainlit.input_widget import Select
from dotenv import load_dotenv
from student_chain import build_student_chain
from evaluator_chain import build_evaluator_chain
from scorer_chain import build_scorer_chain
from models import StudentResponse, TeacherResponse, EvaluatorResponse, ScorerResponse, get_llm
from langchain.memory import ConversationBufferMemory
from typing import Optional
from pathlib import Path
import json
from memory import (
        create_user,
        create_session,
        create_interaction,
        save_student,
        save_teacher,
        save_evaluator,
        save_scorer,
        fetch_session_interaction_ids,
        fetch_student_memory,
)
from qa_generator import generate_qa_for_subtopic, generate_subtopics, load_catalog


load_dotenv()
VS_DIR = VS_DIR = Path(os.getenv("VS_DIR", Path(__file__).resolve().parents[1] / "vectorstore"))
CATALOG_PATH = VS_DIR / "catalog.json"
# ------------------- AUTH ------------------- #


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict,
    default_user: cl.User,
) -> Optional[cl.User]:
    print("Provider: ", provider_id)
    print("Raw user data: ", raw_user_data)
    if provider_id == "google":
        email = raw_user_data.get("email", "")
        if email.endswith("@pilani.bits-pilani.ac.in"):
            return default_user
    return None

# ------------------- ON SETTINGS UPDATE ------------------- #


@cl.on_settings_update
async def setup_agent(settings):
    current_model = settings["Model"]
    print("Updated Model to: ", current_model)
    llm = get_llm(current_model)
    cl.user_session.set("llm", llm)
    user_topic = cl.user_session.get("topic")
    catalog = cl.user_session.get("catalog")
    student_chain, vs = build_student_chain(llm, user_topic, catalog)
    evaluator_chain = build_evaluator_chain(llm)
    scorer_chain = build_scorer_chain(llm)
    cl.user_session.set("student_chain", student_chain)
    cl.user_session.set("evaluator_chain", evaluator_chain)
    cl.user_session.set("scorer_chain", scorer_chain)


# ------------------- ON CHAT RESUME --------------- #
@cl.on_chat_resume
async def on_chat_resume(thread):
    session_memory = ConversationBufferMemory(return_messages=True)
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "user_message":
            session_memory.chat_memory.add_user_message(message["output"])
        else:
            session_memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("session_memory", session_memory)


# ------------------- CHAT START ------------------- #

@cl.on_chat_start
async def start():
    """Initialize session, show topics, generate subtopics, and prepare Q&A."""
    user = cl.user_session.get("user")
    session_id = create_session(user.identifier)
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("session_memory", ConversationBufferMemory(return_messages=True))
    await cl.Message(content=f"Welcome, {user.display_name or user.identifier}!").send()

    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Select Model",
                values=["openai/gpt-4o", "anthropic/claude-3.7-sonnet",
                        "google/gemini-2.5-pro"],
                initial_index=0
            )
        ]
    ).send()
    cl.user_session.set("llm", get_llm(settings["Model"]))

    # Load catalog
    if not CATALOG_PATH.exists():
        await cl.Message(content="⚠️ No catalog found. Please run ingestion first.").send()
        return

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    if not catalog:
        await cl.Message(content="⚠️ Catalog is empty. Add some topics first.").send()
        return

    cl.user_session.set("catalog", catalog)

    # Step 1: Show main topics
    actions = [cl.Action(name=t, payload={"value": t}, label=t) for t in catalog.keys()]
    actions_res = await cl.AskActionMessage(
        content="📚 Here are the available topics. Please choose one:",
        actions=actions
    ).send()

    user_topic = ""
    if actions_res and actions_res.get("payload").get("value"):
        user_topic = actions_res.get("payload").get("value")

    if user_topic not in catalog:
        await cl.Message(content=f"⚠️ '{user_topic}' is not a valid topic. Restart and try again.").send()
        return

    llm = cl.user_session.get("llm")

    # Step 2: Build chains + vectorstore for chosen topic
    student_chain, vs = build_student_chain(llm, user_topic, catalog)
    cl.user_session.set("student_chain", student_chain)

    # Step 3: Generate subtopics (ONE API call)
    subtopics = generate_subtopics(llm, vs)  # returns list of 10 standard subtopics
    cl.user_session.set("subtopics", subtopics)

    # Step 4: Show subtopics to user and let them choose
    subtopic_actions = [cl.Action(name=s, payload={"value": s}, label=s) for s in subtopics]
    subtopic_res = await cl.AskActionMessage(
        content=f"📖 Here are the subtopics for **{user_topic}**. Please choose one:",
        actions=subtopic_actions
    ).send()

    chosen_subtopic = ""
    if subtopic_res and subtopic_res.get("payload").get("value"):
        chosen_subtopic = subtopic_res.get("payload").get("value")

    if chosen_subtopic not in subtopics:
        await cl.Message(content=f"⚠️ '{chosen_subtopic}' is not a valid subtopic. Restart and try again.").send()
        return

    # Step 5: Generate student doubts/questions for the chosen subtopic (ONE API call)
    qa_pool = generate_qa_for_subtopic(llm, vs, chosen_subtopic)  # returns list of QAPair

    # Step 6: Store in session
    cl.user_session.set("qa_pool", qa_pool)
    cl.user_session.set("qa_index", 0)
    cl.user_session.set("topic", user_topic)
    cl.user_session.set("subtopic", chosen_subtopic)
    session_memory = cl.user_session.get("session_memory")

    # Step 7: Kick off conversation
    if qa_pool:
        first_q = qa_pool[0].question
        message = cl.Message(
            content=f"👩‍🎓 Student: Great! Let’s start with **{chosen_subtopic}**. "
                    f"Here’s my first question:\n\n{first_q}"
        )
        await message.send()
        session_memory.chat_memory.add_ai_message(message.content)
    else:
        message = cl.Message(
            content=f"👩‍🎓 Student: I don’t have any questions for {chosen_subtopic} yet."
        )
        await message.send()
        session_memory.chat_memory.add_ai_message(message.content)



# ------------------- MAIN LOOP ------------------- #
@cl.on_message
async def main(message: cl.Message):
    """Handle teacher input, student response, evaluation, and scoring."""
    cl_user = cl.user_session.get("user")  # Chainlit User object
    session_id = cl.user_session.get("session_id")
    session_memory = cl.user_session.get("session_memory")
    if not cl_user:
        await cl.Message(content="❌ User not authenticated.").send()
        return

    model_choice = cl.user_session.get("model", "gemini-1.5-flash")

    print(model_choice)

    # Extract normalized fields
    user_id = cl_user.identifier
    user_email = getattr(cl_user, "email", None)
    user_name = getattr(cl_user, "display_name", None)

    # Ensure user exists in DB
    create_user(user_id, user_email, user_name)

    # Load session state
    student_chain = cl.user_session.get("student_chain")
    evaluator_chain = cl.user_session.get("evaluator_chain")
    scorer_chain = cl.user_session.get("scorer_chain")
    qa_pool = cl.user_session.get("qa_pool", [])
    qa_index = cl.user_session.get("qa_index", 0)
    # Fetch all previous interactions of this session 
    session_interaction_ids = fetch_session_interaction_ids(session_id)

    # Create new interaction entry
    interaction_id = create_interaction(session_id)

    # 1️⃣ Teacher provides explanation
    teacher_explanation = message.content
    teacher_model = TeacherResponse(message=teacher_explanation)
    save_teacher(interaction_id, teacher_model)
    session_memory.chat_memory.add_user_message(teacher_explanation)

    # Expected answer from QA pool
    expected_answer = qa_pool[qa_index].a if qa_index < len(qa_pool) else ""
    student_memory = fetch_student_memory(session_interaction_ids)
    # 2️⃣ Student generates response
    student_llm_response = student_chain.invoke({
        "teacher_explanation": teacher_explanation,
        "student_memory": student_memory
    })

    if isinstance(student_llm_response['text'], StudentResponse):
        student_model = student_llm_response['text']
    else:
        student_model = StudentResponse.parse_raw(student_llm_response['text'])

    save_student(interaction_id, student_model)

    # 3️⃣ Evaluator assesses
    evaluator_llm_response = evaluator_chain.invoke({
        "expected_explanation": expected_answer,
        "teacher_explanation": teacher_explanation,
        "student_question": qa_pool[qa_index].q,
        "student_followup_question": student_model.message,
        "student_response": student_model.json()
    })

    if isinstance(evaluator_llm_response['text'], EvaluatorResponse):
        evaluator_model = evaluator_llm_response['text']
    else:
        evaluator_model = EvaluatorResponse.parse_raw(
            evaluator_llm_response['text'])

    save_evaluator(interaction_id, evaluator_model)


    # 4️⃣ Scorer computes metrics
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

    save_scorer(interaction_id, scorer_model)

    # 5️⃣ Continue conversation
    if student_model.message:
        message = cl.Message(content=f"👩‍🎓 Student: {student_model.message}")
        await message.send()
        session_memory.chat_memory.add_ai_message(message.content)
    else:
        message = cl.Message(content="👩‍🎓 Student: I think I understood this topic.")
        await message.send()
        session_memory.chat_memory.add_ai_message(message.content)
        qa_index += 1
        cl.user_session.set("qa_index", qa_index)
        if qa_index < len(qa_pool):
            next_q = qa_pool[qa_index].q
            await cl.Message(content=f"👩‍🎓 Student: {next_q}").send()
