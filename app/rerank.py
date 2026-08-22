from FlagEmbedding import FlagReranker
import torch
from app.config import RERANK_MODEL_DIR


def load_reranker():
    if not RERANK_MODEL_DIR.exists():
        raise FileNotFoundError(f'Invalid Reranker model path: {RERANK_MODEL_DIR}')
    return FlagReranker(
        str(RERANK_MODEL_DIR),
        use_fp16=torch.cuda.is_available()
    )

def rerank_docs(query, hybrid_docs, reranker, top_k=5):
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not hybrid_docs:
        return []

    pairs = []

    for result in hybrid_docs:
        title = result.doc.metadata.get("title", "")
        content = result.doc.page_content
        text = f"{title}\n{content}"

        pairs.append([query, text])

    scores = reranker.compute_score(
        pairs,
        normalize=True
    )

    if len(scores) != len(hybrid_docs):
        raise ValueError("Mismatch between scores and hybrid docs.")

    for result, score in zip(hybrid_docs, scores):
        result.rerank_score = float(score)

    sorted_docs = sorted(
        hybrid_docs,
        key=lambda x: x.rerank_score,
        reverse=True
    )

    return sorted_docs[:top_k]


def unique_source(reranked_docs):
    unique_docs = {}
    for result in reranked_docs:
        doc_id = result.doc.metadata.get("doc_id")
        if doc_id and doc_id not in unique_docs:
            unique_docs[doc_id] = result.doc
    return list(unique_docs.values())