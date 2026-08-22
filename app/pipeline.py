from app.chunk_data import chunk_docs, load_data
from app.genarate import generator, get_llm
from app.hybrid_search import build_bm25, hybrid_search
from app.logger import get_logger
from app.rerank import load_reranker, rerank_docs,unique_source
from app.vector_store import get_vector_store


logger = get_logger(__name__)


def load_models():
    stage = "load reranker"
    try:
        reranker = load_reranker()
        logger.info("Reranker loaded successfully")

        stage = "load llm"
        llm = get_llm()
        logger.info("LLM loaded successfully")

        return reranker, llm
    except Exception as exc:
        logger.exception("Model loading failed at %s", stage)
        raise RuntimeError(f"Model loading failed during {stage}: {exc}") from exc


def build_index():
    stage = "load data"
    try:
        logger.info("Starting index build")

        data = load_data()
        logger.info("Data loaded successfully")

        stage = "chunk documents"
        docs = chunk_docs(data)
        logger.info("Chunking completed: %s chunks", len(docs))

        stage = "build vector store"
        vector_store = get_vector_store(docs)
        logger.info("Vector store created successfully")

        stage = "build bm25"
        bm25 = build_bm25(docs)
        logger.info("BM25 created successfully")

        return vector_store, bm25, docs
    except Exception as exc:
        logger.exception("System build failed at %s", stage)
        raise RuntimeError(f"System build failed during {stage}: {exc}") from exc


def run_pipeline(query, vector_store, bm25, reranker, llm, docs):
    if not isinstance(query, str):
        raise ValueError("Query must be a string.")

    query_text = query.strip()

    if not query_text:
        raise ValueError("Query cannot be empty.")

    stage = "hybrid search"

    try:
        logger.info("Pipeline started: %s", query_text)

        hybrid_docs = hybrid_search(query_text, vector_store, docs, bm25)
        logger.info("Hybrid search result: %s", len(hybrid_docs))

        if not hybrid_docs:
            logger.warning("No documents found for query: %s", query_text)
            return "No relevant documents found", []

        stage = "rerank"
        reranked_docs = rerank_docs(query_text, hybrid_docs, reranker)
        logger.info("Reranked docs: %s", len(reranked_docs))

        stage = "select top docs"
        sources = unique_source(reranked_docs)
        logger.info("Unique sources: %s", len(sources))

        if not sources:
            logger.warning("No unique sources found for query: %s", query_text)
            return "No relevant documents found", []

        stage = "generate answer"
        answer = generator(query_text, reranked_docs, llm)
        logger.info("Answer generated successfully")
        return answer, sources
    except Exception as exc:
        logger.exception("Pipeline failed at %s for query: %s", stage, query_text)
        raise RuntimeError(f"Pipeline failed during {stage} for query '{query_text}': {exc}") from exc
