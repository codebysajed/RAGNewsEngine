import hashlib
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import DATA_DIR
from app.logger import get_logger


logger = get_logger(__name__)


def load_data(path=DATA_DIR):
    data_path = Path(path)

    try:
        with data_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Archive file not found at {data_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in archive file {data_path} at line {exc.lineno}, column {exc.colno}"
        ) from exc
    except OSError as exc:
        raise OSError(f"Could not read archive file {data_path}: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(
            f"Archive file {data_path} must contain a list of articles, got {type(data).__name__}"
        )

    return data

def normalize_screenshot_path(screenshot):
    if not isinstance(screenshot, str) or not screenshot.strip():
        return ""

    return f"screenshots/{Path(screenshot.replace('\\', '/')).name}"


def make_doc_id(url):
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Article URL is missing.")
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


def chunk_docs(data):
    if not data:
        raise ValueError("No article data found for chunking.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    docs = []

    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            logger.warning("Skipping item %s because it is %s", index, type(item).__name__)
            continue

        latest = item.get("latest_change") or {}
        if not isinstance(latest, dict):
            logger.warning("Skipping item %s because latest_change is %s", index, type(latest).__name__)
            continue

        title = latest.get("current_title") or item.get("title", "")
        content = latest.get("current_content_excerpt") or item.get("content", "")
        publish_date = latest.get("current_publish_date") or item.get("publish_date", "")

        deleted = bool(item.get("deleted", False))
        change_count = item.get("change_count", 0)

        try:
            change_count = int(change_count)
        except (TypeError, ValueError):
            change_count = 0

        updated = change_count > 0 and not deleted

        url = item.get("url", "")
        author = item.get("author", "")
        source = item.get("source", "")
        screenshot = normalize_screenshot_path(item.get("screenshot", ""))
    


        if not isinstance(url, str) or not url.strip():
            logger.warning("Skipping article '%s' at index %s because url is missing", title or "unknown", index)
            continue

        try:
            doc_id = make_doc_id(url)
        except ValueError as exc:
            logger.warning("Skipping article '%s' at index %s: %s", title or "unknown", index, exc)
            continue

        text = f"{title}\n\n{content}"
        if not text.strip():
            logger.warning("Skipping article '%s' at index %s because text is empty", title or "unknown", index)
            continue

        chunks = splitter.split_text(text)
        chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 100]

        if not chunks:
            logger.warning(
                "Skipping article '%s' at index %s because no chunk met minimum length",
                title or "unknown",
                index,
            )
            continue

        for chunk_index, chunk in enumerate(chunks, start=1):
            metadata = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{chunk_index}",
                "chunk_index": chunk_index,
                "title": title,
                "url": None if deleted else url,
                "author": author,
                "source": source,
                "publish_date": publish_date,
                "screenshot": screenshot,
            }

            if deleted:
                metadata["status_note"] = item.get("status_note", "Deleted by publisher")
            elif updated:
                metadata["status_note"] = item.get("status_note", "Updated by publisher")
                metadata["changed_fields"] = latest.get("changed_fields", [])
                metadata["previous_title"] = latest.get("previous_title", "")
                metadata["previous_publish_date"] = latest.get("previous_publish_date", "")
                metadata["previous_screenshot"] = normalize_screenshot_path(latest.get("previous_screenshot", screenshot))

            docs.append(
                Document(
                    page_content=chunk,
                    metadata=metadata,
                )
            )

    if not docs:
        raise ValueError("No chunks were generated from the provided article data.")

    return docs
