import chainlit as cl
from dotenv import load_dotenv

from langchain.output_parsers import PydanticOutputParser
from memory import load_memory, save_interaction
from student_chain import build_student_chain
from qa_generator import generate_initial_qa
from models import StudentResponse, TeacherResponse
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

    # Generate initial Q&A pool
    qa_pool = generate_initial_qa(vs, n=5)
    cl.user_session.set("qa_pool", qa_pool)
    cl.user_session.set("qa_index", 0)
    cl.user_session.set("student_chain", student_chain)
    cl.user_session.set("student_memory", student_memory)

    # Student initiates conversation
    if qa_pool:
        first_q = qa_pool[0].q
        await cl.Message(content=f"👩‍🎓 Student: {first_q}").send()
    else:
        await cl.Message(content="👩‍🎓 Student: I don’t have any questions yet.").send()


@cl.on_message
async def main(message: cl.Message):
    """Handle teacher responses and student follow-up."""
    student_chain = cl.user_session.get("student_chain")
    # student_memory = cl.user_session.get("student_memory")
    student_memory = load_memory("student")
    qa_pool = cl.user_session.get("qa_pool", [])
    qa_index = cl.user_session.get("qa_index", 0)

    # Teacher provides explanation
    teacher_explanation = message.content
    teacher_model = TeacherResponse(message=teacher_explanation)
    interaction_id = save_interaction(
        "teacher", teacher_model, interaction_id=None)
    expected_answer = qa_pool[qa_index].a if qa_index < len(qa_pool) else ""
    # Student generates response using chain
    student_llm_response = student_chain.invoke({
        "student_memory": student_memory,
        "teacher_explanation": teacher_explanation
    })

    print("==========STUDENT RESPONSE START==================")
    print(student_llm_response['text'])
    print("==========STUDENT RESPONSE END==================")

    if isinstance(student_llm_response['text'], StudentResponse):
        student_model = student_llm_response['text']
    else:
        student_model = StudentResponse.parse_raw(student_llm_response['text'])
    save_interaction("student", student_model, interaction_id)

    # Send follow-up if needed
    student_response = StudentResponse.parse_obj(student_llm_response['text'])
    if student_response.message:
        await cl.Message(content=f"👩‍🎓 Student: {student_response.message}").send()
    else:
        await cl.Message(content="👩‍🎓 Student: I think I understood this topic.").send()
        # Move to next QA pool question
        qa_index += 1
        cl.user_session.set("qa_index", qa_index)
        if qa_index < len(qa_pool):
            next_q = qa_pool[qa_index].q
            await cl.Message(content=f"👩‍🎓 Student: {next_q}").send()
