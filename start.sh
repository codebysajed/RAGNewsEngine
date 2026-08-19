#!/bin/sh

set -e

echo "========================================"
echo "Starting RAG News Backend"
echo "========================================"


# ----------------------------------------
# Embedding Model
# ----------------------------------------

EMBEDDING_MODEL="/app/models/bengali-sentence-similarity-sbert"

if [ ! -f "$EMBEDDING_MODEL/config.json" ]; then

    echo "Embedding model not found."
    echo "Downloading from Hugging Face..."

    python -c "
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='l3cube-pune/bengali-sentence-similarity-sbert',
    local_dir='$EMBEDDING_MODEL'
)
"

    echo "Embedding model downloaded successfully."

else

    echo "Embedding model already exists."

fi


# ----------------------------------------
# Reranker Model
# ----------------------------------------

RERANKER_MODEL="/app/models/bge-reranker-v2-m3"

if [ ! -f "$RERANKER_MODEL/config.json" ]; then

    echo "Reranker model not found."
    echo "Downloading from Hugging Face..."

    python -c "
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='BAAI/bge-reranker-v2-m3',
    local_dir='$RERANKER_MODEL'
)
"

    echo "Reranker model downloaded successfully."

else

    echo "Reranker model already exists."

fi


# ----------------------------------------
# Start FastAPI
# ----------------------------------------

echo "========================================"
echo "Starting FastAPI..."
echo "========================================"

exec uvicorn app.api:app \
    --host 0.0.0.0 \
    --port 8000