from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "storage" / "news" / "archive_articles.json"
EMBEDDING_MODEL_DIR = BASE_DIR / "models" / "bengali-sentence-similarity-sbert"
RERANK_MODEL_DIR = BASE_DIR / "models" / "bge-reranker-v2-m3"
INDEX_DIR = BASE_DIR / "vector_store"
INDEX_FILE = INDEX_DIR / "index.faiss"
PKL_FILE = INDEX_DIR/ "index.pkl"
META_FILE = INDEX_DIR/ "metadata.json"

