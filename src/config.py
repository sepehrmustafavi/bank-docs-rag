from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "bank_docs"

EMBED_MODEL = "intfloat/multilingual-e5-base"
LLM_MODEL = "llama3.1:8b"

CHUNK_SIZE = 500      
CHUNK_OVERLAP = 100  
TOP_K = 3           