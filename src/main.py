import chainlit as cl
from dotenv import load_dotenv

from memory import load_memory, save_interaction
from student_chain import build_student_chain


@cl.on_chat_start
async def on_start():
    load_dotenv()
    memory = load_memory()
    cl.user_session.set("memory", memory)

    chain, vs = build_student_chain()
    cl.user_session.set("chain", chain)
    cl.user_session.set("vs", vs)

    await cl.Message(content="👩‍🏫 Welcome! Teach me something and I'll respond as your student.").send()


@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")
    chain = cl.user_session.get("chain")
    vs = cl.user_session.get("vs")

    teacher_explanation = message.content

    # Retrieve context
    docs = vs.similarity_search(teacher_explanation, k=3)
    context = "\n\n".join(d.page_content for d in docs)

    # Run student response
    result = await chain.arun(context=context, teacher_explanation=teacher_explanation)

    # Save interaction
    save_interaction(memory, teacher_explanation, result)

    await cl.Message(content=f"📘 Student:\n{result}").send()
