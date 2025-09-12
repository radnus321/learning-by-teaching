import os
import json
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pathlib import Path


load_dotenv()
VS_DIR = os.getenv("VS_DIR")
CATALOG_PATH = Path(VS_DIR + "/catalog.json")


def load_catalog():
    if CATALOG_PATH.exists():
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


class QAPair(BaseModel):
    q: str
    a: str


class QAList(BaseModel):
    questions: List[QAPair]


def generate_initial_qa(vs, n: int = 10) -> List[QAPair]:
    """
    Generate initial Q&A pairs from vectorDB content.

    Args:
        vs: Vectorstore instance (e.g., Chroma).
        n: Number of context documents to retrieve.

    Returns:
        List[QAPair]: A list of Q&A pairs as Pydantic objects.
    """
    # Pull relevant documents
    docs = vs.similarity_search("learning by teaching", k=n)
    context = "\n\n".join(d.page_content for d in docs)

    # Pydantic parser
    parser = PydanticOutputParser(pydantic_object=QAList)

    # Prompt template
    template = """
    You are a curious student preparing questions about "learning by teaching".
    - Based on the following textbook context, generate a list of natural student
    questions and their accurate answers.
    - Only ask questions which are technical and require conceptual knowledge.
    - ** DO NOT ASK QUESTIONS WHICH REQUIRE MEMORIZATION OF THE MATERIAL **

    Each answer should be concise, factually correct, and directly address the question.

    Respond ONLY in valid JSON following this schema:
    {format_instructions}

    Context:
    {context}
    """

    # Build LLM chain
    prompt = ChatPromptTemplate.from_template(template).partial(
        format_instructions=parser.get_format_instructions()
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4)
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run
    result = chain.run(context=context)

    try:
        parsed = parser.parse(result)
        return parsed.questions
    except Exception as e:
        print("Parsing failed:", e)
        return []
