from app.config import EMBEDDING_MODEL_DIR
from langchain_huggingface import HuggingFaceEmbeddings
import torch


def get_embedding():
    if not EMBEDDING_MODEL_DIR.exists():
        raise FileNotFoundError('Invalid Embedding model path')
    return HuggingFaceEmbeddings(
        model_name = str(EMBEDDING_MODEL_DIR),
        model_kwargs = {"device":"cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs = {"normalize_embeddings":True}
    )
