from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
VS_DIR = BASE_DIR / "vectorstore"
MEMORY_FILE = BASE_DIR / "chat_memory.json"
