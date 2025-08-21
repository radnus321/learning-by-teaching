from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from config import VS_DIR


def build_student_chain():
    """Return (LLM chain, vectorstore)."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vs = Chroma(persist_directory=str(VS_DIR), embedding_function=embeddings)

    template = """
    You are a curious student. 
    The user is your teacher, explaining a concept.
    You also have access to some textbook context.

    Context from textbook:
    {context}

    Teacher just said:
    "{teacher_explanation}"

    Your job:
    - Ask 1–2 clarifying questions (as a curious student).
    - Point out missing details if any.
    - Rate the explanation as "Good", "Partial", or "Needs Work".

    Respond ONLY in valid JSON.
    {{
      "questions": ["..."],
      "missing_points": ["..."],
      "rating": "Good|Partial|Needs Work"
    }}
    """
    prompt = ChatPromptTemplate.from_template(template)
    return LLMChain(llm=llm, prompt=prompt, verbose=True), vs
