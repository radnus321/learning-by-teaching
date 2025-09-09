from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from models import StudentResponse

student_prompt = """
You are a curious student learning from the teacher. You do not have any prior knowledge of the topic other than the Student Context provided below.

Student Context:
{student_memory}

Teacher explanation:
"{teacher_explanation}"

Your task:
- Decide how well you understood the explanation.
  * Fully clear → rating = "understood", message = null
  * Partially clear → rating = "needs work", ask ONE concise follow-up question
  * Confused → rating = "confused", ask ONE clarifying question
- Write a short reflection about your understanding in natural language, e.g., "I didn't understand sorting properly."
- List missing points you noticed, if any.

Respond ONLY in valid JSON that matches this schema:
{{
  "message": "Your follow-up question or null",
  "rating": "understood|needs work|confused",
  "reflection": "How you understood the concept",
  "missing_points": ["point1", "point2"]
}}
"""


def build_student_chain(topic: str, catalog: dict):
    """Return (LLM chain, vectorstore) for a given topic using catalog path."""
    # LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4)

    # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Get vectorstore path from catalog
    topic_info = catalog.get(topic)
    if not topic_info:
        raise ValueError(f"Topic '{topic}' not found in catalog")

    vs_path = topic_info["vectorstore_path"]

    # Load Chroma for this topic
    vs = Chroma(
        persist_directory=vs_path,
        embedding_function=embeddings
    )

    # Parser
    parser = PydanticOutputParser(pydantic_object=StudentResponse)

    # Prompt with parser format instructions
    prompt = ChatPromptTemplate.from_template(student_prompt).partial(
        format_instructions=parser.get_format_instructions()
    )

    # Chain
    chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)

    return chain, vs
