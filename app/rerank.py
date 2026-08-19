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

def rerank_docs(query, hybrid_docs, reranker, threshold=0.70):
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

    filtered_docs = [
        result
        for result in sorted_docs
        if result.rerank_score >= threshold
    ]

    if filtered_docs:
        return filtered_docs

    return sorted_docs


def deduplication(docs):
    if not docs:
        raise ValueError("Docs are empty.")

    best_docs = {}

    for result in docs:
        doc_id = result.doc.metadata.get("doc_id")
        score = result.rerank_score

        if doc_id not in best_docs:
            best_docs[doc_id] = result

        elif score > best_docs[doc_id].rerank_score:
            best_docs[doc_id] = result

    return list(best_docs.values())


def select_top_docs(docs, k=5):
    if not docs:
        raise ValueError("Docs are empty.")

    dedup_docs = deduplication(docs)

    sorted_docs = sorted(
        dedup_docs,
        key=lambda x: x.rerank_score,
        reverse=True
    )

    return sorted_docs[:k]