from rank_bm25 import BM25Okapi
import numpy as np
from langchain_core.documents import Document
from collections import defaultdict
from dataclasses import dataclass

def semantic_search(query, db, k):
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    results = db.similarity_search_with_score(query,k)

    scores = {}

    for doc, score in results:
        chunk_id = doc.metadata.get('chunk_id')
        if chunk_id:
            scores[chunk_id] = score

    return scores

def build_bm25(docs):
    if not docs:
        raise ValueError('Not found any docs')
    tokenized_texts = [doc.page_content.lower().split() for doc in docs]
    return BM25Okapi(tokenized_texts)

def bm25_search(query,docs,bm25, k):
    if not query.strip():
        raise ValueError('Query can not be empty')
    results = bm25.get_scores(query.lower().split())

    bm_scores = {}

    for doc, score in zip(docs,results):
        chunk_id = doc.metadata.get('chunk_id')

        if chunk_id:
            bm_scores[chunk_id] = np.log1p(score)

    max_score = max(bm_scores.values(), default=0)

    if max_score > 0:
        for chunk_id in bm_scores:
            bm_scores[chunk_id]/=max_score

    sorted_score = sorted(
        bm_scores.items(),
        key= lambda x : x[1],
        reverse=True

    )

    top_scores = {}
    for chunk_id, score in sorted_score[:k]:
        top_scores[chunk_id] = score

    return top_scores



@dataclass
class searchresult():
    doc:Document
    hybrid_score : float
    rerank_score : float=0.0



def hybrid_search(query,db,docs,bm25,sema_weight = 0.65, bm25_weight = 0.25,k=20):

    semantic_results = semantic_search(query,db,k)
    bm25_results = bm25_search(query,docs,bm25,k)

    combine_score = defaultdict(float)

    for chunk_id, score in semantic_results.items():
        combine_score[chunk_id] += score * sema_weight
    for chunk_id, score in bm25_results.items():
        combine_score[chunk_id] += score * bm25_weight

    sortede_score = sorted(
        combine_score.items(),
        key=lambda x : x[1],
        reverse=True
    )[:k]

    doc_map = {doc.metadata.get('chunk_id'):doc for doc in docs if doc.metadata.get('chunk_id')}
    final_docs = []
    for chunk_id,  score in sortede_score:
        doc = doc_map[chunk_id]
        if doc:
            final_docs.append(searchresult(
                doc = doc,
                hybrid_score=score
            ))
    return final_docs
