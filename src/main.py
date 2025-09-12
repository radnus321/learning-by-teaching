import os
from db import (
    users_collection,
    interaction_collection,
    teacher_collection,
    student_collection,
    evaluator_collection,
    scorer_collection,
)
from datetime import datetime
import uuid
import chainlit as cl
from chainlit.input_widget import Select
from dotenv import load_dotenv
from student_chain import build_student_chain
from evaluator_chain import build_evaluator_chain
from scorer_chain import build_scorer_chain
from qa_generator import generate_initial_qa, load_catalog
from models import StudentResponse, TeacherResponse, EvaluatorResponse, ScorerResponse, get_llm
from typing import Optional
from pathlib import Path
import json

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


# ------------------- CHAT START ------------------- #


@cl.on_chat_start
async def start():
    """Initialize session, show topics, and prepare Q&A after topic selection."""
    user = cl.user_session.get("user")
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

    # Step 1: Student says hi
    await cl.Message(content="👩‍🎓 Student: Hi! I’m ready to learn.").send()

    # Step 2: Show topics as buttons
    actions = [
        cl.Action(name=t, payload={"value": t}, label=t) for t in catalog.keys()]
    actions_res = await cl.AskActionMessage(
        content="📚 Here are the available topics. Please choose one:",
        actions=actions
    ).send()

    user_topic = ""
    if actions_res and actions_res.get("payload").get("value"):
        user_topic = actions_res.get("payload").get("value")

    catalog = cl.user_session.get("catalog")

    if user_topic not in catalog:
        await cl.Message(content=f"⚠️ '{user_topic}' is not a valid topic. Restart and try again.").send()
        return

    llm = cl.user_session.get("llm")

    # Step 3: Build chains + vectorstore for chosen topic
    student_chain, vs = build_student_chain(llm, user_topic, catalog)
    evaluator_chain = build_evaluator_chain(llm)
    scorer_chain = build_scorer_chain(llm)

    # Step 4: Generate initial Q&A immediately
    qa_pool = generate_initial_qa(vs, n=5)

    # Step 5: Store in session
    cl.user_session.set("student_chain", student_chain)
    cl.user_session.set("evaluator_chain", evaluator_chain)
    cl.user_session.set("scorer_chain", scorer_chain)
    cl.user_session.set("qa_pool", qa_pool)
    cl.user_session.set("qa_index", 0)
    cl.user_session.set("topic", user_topic)

    # Step 6: Kick off conversation
    if qa_pool:
        first_q = qa_pool[0].q
        await cl.Message(
            content=f"👩‍🎓 Student: Great! Let’s start with **{user_topic}**. "
            f"Here’s my first question:\n\n{first_q}"
        ).send()
    else:
        await cl.Message(
            content=f"👩‍🎓 Student: I don’t have any questions for {
                user_topic} yet."
        ).send()


# ------------------- MAIN LOOP ------------------- #
@cl.on_message
async def main(message: cl.Message):
    """Handle teacher input, student response, evaluation, and scoring."""
    cl_user = cl.user_session.get("user")  # Chainlit User object
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
    users_collection.update_one(
        {"_id": user_id},
        {"$setOnInsert": {
            "_id": user_id,
            "email": user_email,
            "name": user_name,
            "created_at": datetime.utcnow()
        }},
        upsert=True
    )

    # Load session state
    student_chain = cl.user_session.get("student_chain")
    evaluator_chain = cl.user_session.get("evaluator_chain")
    scorer_chain = cl.user_session.get("scorer_chain")
    qa_pool = cl.user_session.get("qa_pool", [])
    qa_index = cl.user_session.get("qa_index", 0)
    # Fetch all previous interactions of this user
    user_interactions = interaction_collection.find(
        {"user_id": user_id},
        {"_id": 1}  # we only need the interaction IDs
    )
    interaction_ids = [i["_id"] for i in user_interactions]

    # Create new interaction entry
    interaction_id = str(uuid.uuid4())
    interaction_collection.insert_one({
        "_id": interaction_id,
        "user_id": user_id,
        "timestamp": datetime.utcnow()
    })

    # 1️⃣ Teacher provides explanation
    teacher_explanation = message.content
    teacher_model = TeacherResponse(message=teacher_explanation)
    teacher_collection.insert_one({
        "_id": interaction_id,
        **teacher_model.dict(),
        "timestamp": datetime.utcnow()
    })

    # Expected answer from QA pool
    expected_answer = qa_pool[qa_index].a if qa_index < len(qa_pool) else ""
    # Fetch all student responses for these interactions
    student_memory_docs = student_collection.find(
        {"_id": {"$in": interaction_ids}}
    )

    # Convert to a list of dicts or models
    student_memory = [StudentResponse(**doc) for doc in student_memory_docs]
    # 2️⃣ Student generates response
    student_llm_response = student_chain.invoke({
        "teacher_explanation": teacher_explanation,
        "student_memory": student_memory
    })

    if isinstance(student_llm_response['text'], StudentResponse):
        student_model = student_llm_response['text']
    else:
        student_model = StudentResponse.parse_raw(student_llm_response['text'])

    student_collection.insert_one({
        "_id": interaction_id,
        **student_model.dict(),
        "timestamp": datetime.utcnow()
    })

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

    evaluator_collection.insert_one({
        "_id": interaction_id,
        **evaluator_model.dict(),
        "timestamp": datetime.utcnow()
    })

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

    scorer_collection.insert_one({
        "_id": interaction_id,
        **scorer_model.dict(),
        "timestamp": datetime.utcnow()
    })

    # 5️⃣ Continue conversation
    if student_model.message:
        await cl.Message(content=f"👩‍🎓 Student: {student_model.message}").send()
    else:
        await cl.Message(content="👩‍🎓 Student: I think I understood this topic.").send()
        qa_index += 1
        cl.user_session.set("qa_index", qa_index)
        if qa_index < len(qa_pool):
            next_q = qa_pool[qa_index].q
            await cl.Message(content=f"👩‍🎓 Student: {next_q}").send()
