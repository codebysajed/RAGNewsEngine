from app.config import EMBEDDING_MODEL_DIR,INDEX_DIR, INDEX_FILE, PKL_FILE, META_FILE
from app.embedding import get_embedding
import json
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS


def build_vectordb(docs):
    if not docs:
        raise ValueError("No data found for vector database.")

    embedding = get_embedding()

    db = FAISS.from_documents(
        docs,
        embedding,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    db.save_local(str(INDEX_DIR))

    metadata = {
        "embedding_model": str(EMBEDDING_MODEL_DIR),
        "distance_strategy": "MAX_INNER_PRODUCT",
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    return db


def load_vector():
    required = [INDEX_DIR, INDEX_FILE, PKL_FILE, META_FILE]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if metadata.get("embedding_model") != str(EMBEDDING_MODEL_DIR):
        raise ValueError(
            f"Embedding model mismatch. Expected: {EMBEDDING_MODEL_DIR}"
        )

    if metadata.get("distance_strategy") != "MAX_INNER_PRODUCT":
        raise ValueError(
            "Distance strategy mismatch. Expected: MAX_INNER_PRODUCT"
        )

    embedding = get_embedding()

    return FAISS.load_local(
        str(INDEX_DIR),
        embedding,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        allow_dangerous_deserialization=True,
    )

def get_vector_store(docs):
    if not docs:
        raise ValueError("No documents found for vector store.")

    if INDEX_FILE.exists() and PKL_FILE.exists():
        return load_vector()

    return build_vectordb(docs)