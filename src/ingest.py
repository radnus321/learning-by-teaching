import os
import glob
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
VS_DIR = Path(__file__).resolve().parents[1] / "vectorstore"


def load_documents():
    """Load PDFs, TXTs, and MDs from data/ folder."""
    docs = []
    pdf_paths = glob.glob(str(DATA_DIR / "*.pdf"))
    txt_paths = glob.glob(str(DATA_DIR / "*.txt"))
    md_paths = glob.glob(str(DATA_DIR / "*.md"))

    if not (pdf_paths or txt_paths or md_paths):
        print(f"[ingest] No source files found in {DATA_DIR}. Add PDF/TXT/MD files and rerun.")
        return []

    for p in pdf_paths:
        try:
            docs.extend(PyPDFLoader(p).load())
        except Exception as e:
            print(f"[ingest] Failed to load PDF {p}: {e}")

    for p in txt_paths + md_paths:
        try:
            docs.extend(TextLoader(p, encoding="utf-8").load())
        except Exception as e:
            print(f"[ingest] Failed to load text {p}: {e}")

    return docs


def split_docs(docs):
    """Split into manageable chunks for embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(docs)


def build_vectorstore(splits):
    """Embed chunks with Gemini and store in Chroma."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vs = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(VS_DIR),
    )
    return vs


def main():
    load_dotenv()
    print(f"[ingest] DATA_DIR={DATA_DIR}")
    print(f"[ingest] VS_DIR={VS_DIR}")

    docs = load_documents()
    if not docs:
        return

    print(f"[ingest] Loaded {len(docs)} documents")
    splits = split_docs(docs)
    print(f"[ingest] Split into {len(splits)} chunks")

    _ = build_vectorstore(splits)
    print("[ingest] Vector store built and persisted.")


if __name__ == "__main__":
    main()
