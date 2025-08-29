import chainlit as cl
from dotenv import load_dotenv

from memory import load_memory, save_interaction
from student_chain import build_student_chain
from qa_generator import generate_initial_qa
from student_chain import StudentResponse

load_dotenv()

# simulator to simulate student and teacher bot
# evaluator which evaluates the responses --> first
# genetic/evolutionary algo to generate syntethic data to be used in simulator


@cl.on_chat_start
async def start():
    """Initialize memory, chain, and seed questions when chat starts."""
    memory = load_memory()
    chain, vs = build_student_chain()

    # Generate initial Q&A pool
    qa_pool = generate_initial_qa(vs, n=5)
    cl.user_session.set("qa_pool", qa_pool)
    cl.user_session.set("qa_index", 0)
    cl.user_session.set("chain", chain)
    cl.user_session.set("memory", memory)

    # Student initiates conversation
    if qa_pool:
        first_q = qa_pool[0].q
        await cl.Message(content=f"👩‍🎓 Student: {first_q}").send()
    else:
        await cl.Message(content="👩‍🎓 Student: I don’t have any questions yet.").send()


@cl.on_message
async def main(message: cl.Message):
    """Handle teacher responses and student follow-up."""
    chain = cl.user_session.get("chain")
    memory = cl.user_session.get("memory")
    qa_pool = cl.user_session.get("qa_pool", [])
    qa_index = cl.user_session.get("qa_index", 0)

    teacher_explanation = message.content
    expected_answer = qa_pool[qa_index].a if qa_index < len(qa_pool) else ""

    # Student generates response using chain
    result = chain.invoke({
        "context": expected_answer,
        "teacher_explanation": teacher_explanation
    })

    # Directly parse into Pydantic model
    print(result.get("text"))
    student_response = StudentResponse.parse_obj(result.get("text"))

    # Send follow-up if needed
    if student_response.question:
        await cl.Message(content=f"👩‍🎓 Student: {student_response.question}").send()
    else:
        await cl.Message(content="👩‍🎓 Student: I think I understood this topic.").send()

        # Save interaction
        save_interaction(memory, teacher_explanation, student_response)

        # Move to next QA pool question
        qa_index += 1
        cl.user_session.set("qa_index", qa_index)
        if qa_index < len(qa_pool):
            next_q = qa_pool[qa_index].q
            await cl.Message(content=f"👩‍🎓 Student: {next_q}").send()
