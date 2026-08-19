import html
import json
import os
from pathlib import Path

import requests
import streamlit as st
from PIL import Image, ImageFile


# ============================================================
# Configuration
# ============================================================

DEFAULT_BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000/ask",
)

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# Utility
# ============================================================

def rerun_page():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def ask_backend(url: str, query: str):
    response = requests.post(
        url,
        json={"query": query},
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


def text_value(value):
    if value is None:
        return ""

    if isinstance(value, (list, tuple, set)):
        items = [
            text_value(item)
            for item in value
            if item not in (None, "", [], {})
        ]
        return ", ".join(
            item for item in items if item
        )

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    return str(value).strip()


# ============================================================
# Source helpers
# ============================================================

def source_title(source, index):
    if not isinstance(source, dict):
        return f"Source {index}"

    for key in ("title", "headline", "name"):
        value = text_value(source.get(key))

        if value:
            return value

    return f"Source {index}"


# ============================================================
# Native Metadata Card
# ============================================================

def render_metadata_field(label, value, link=False):
    value = text_value(value)

    if not value:
        return

    st.markdown(
        f"**{label}**"
    )

    if link and value.startswith(
        ("http://", "https://")
    ):
        st.markdown(
            f"[Open article]({value})"
        )
    else:
        st.write(value)


def render_metadata_fields(source):
    """
    Metadata is rendered using native Streamlit components.
    No raw HTML is used here.
    """

    fields = [
        ("Source", "source", False),
        ("Author", "author", False),
        ("Published", "publish_date", False),
        ("Status Note", "status_note", False),
        ("Chunk Index", "chunk_index", False),
        ("Article URL", "url", True),
        ("Changed Fields", "changed_fields", False),
        ("Previous Title", "previous_title", False),
        (
            "Previous Publish Date",
            "previous_publish_date",
            False,
        ),
    ]

    for start in range(0, len(fields), 2):

        col1, col2 = st.columns(
            2,
            gap="large",
        )

        first = fields[start]

        with col1:
            render_metadata_field(
                first[0],
                source.get(first[1]),
                link=first[2],
            )

        if start + 1 < len(fields):

            second = fields[start + 1]

            with col2:
                render_metadata_field(
                    second[0],
                    source.get(second[1]),
                    link=second[2],
                )

        st.divider()


# ============================================================
# Screenshot
# ============================================================

def resolve_image_path(raw_value):
    """
    Backend may return:

    screenshots/file.png

    D:\\News\\storage\\screenshots\\file.png

    /app/storage/screenshots/file.png

    This function converts all of them to the
    frontend Docker path.
    """

    path = text_value(raw_value)

    if not path:
        return None

    normalized = path.replace("\\", "/").strip()

    filename = Path(normalized).name

    if not filename:
        return None

    # ========================================================
    # Docker frontend path
    # ========================================================

    docker_path = (
        Path("/app/storage/screenshots")
        / filename
    )

    try:
        if (
            docker_path.exists()
            and docker_path.is_file()
        ):
            return docker_path
    except OSError:
        pass

    # ========================================================
    # Local development fallback
    # ========================================================

    candidates = [
        Path(path).expanduser(),
        Path.cwd() / normalized,
        Path(__file__).resolve().parent / normalized,
    ]

    for candidate in candidates:

        try:
            if (
                candidate.exists()
                and candidate.is_file()
            ):
                return candidate

        except OSError:
            continue

    return None


def load_preview_image(image_path):
    try:

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            image.thumbnail(
                (2000, 1500)
            )

            return image.copy()

    except Exception as exc:

        st.warning(
            f"Screenshot preview unavailable: {exc}"
        )

        return None


def render_screenshot_preview(source):

    if not isinstance(source, dict):
        return

    screenshot = source.get("screenshot")

    if not screenshot:
        return

    image_path = resolve_image_path(
        screenshot
    )

    if image_path is None:

        st.warning(
            "Screenshot file not found."
        )

        return

    st.subheader("Screenshot")

    preview = load_preview_image(
        image_path
    )

    if preview is not None:

        st.image(
            preview,
            caption="Screenshot",
            width="stretch",
        )


# ============================================================
# Answer
# ============================================================

def render_answer(answer):

    answer_text = text_value(answer)

    if not answer_text:
        return

    st.subheader("Answer")

    st.write(answer_text)


# ============================================================
# Metadata
# ============================================================

def render_metadata(source, index):

    if not isinstance(source, dict):
        return

    st.markdown(
        f"## Source {index}"
    )

    title = source_title(
        source,
        index,
    )

    st.markdown(
        f"### {title}"
    )

    # ========================================================
    # Metadata + Screenshot
    # ========================================================

    left_col, right_col = st.columns(
        [1.45, 1],
        gap="large",
    )

    with left_col:

        st.subheader("Metadata")

        render_metadata_fields(
            source
        )

    with right_col:

        render_screenshot_preview(
            source
        )

    st.divider()


# ============================================================
# CSS
# ============================================================

def setup_styles():

    st.markdown(
        """
        <style>

        .stApp {
            background:
                linear-gradient(
                    180deg,
                    #f7f8fa 0%,
                    #edf2f7 100%
                );
            color: #000000 !important;
        }

        .block-container {
            max-width: 1100px;
            padding-top: 1rem;
            padding-bottom: 2rem;
        }

        /* All normal text */
        .stApp,
        .stApp p,
        .stApp span,
        .stApp label,
        .stApp div,
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6 {
            color: #000000;
        }

        /* Metadata text */
        [data-testid="stMarkdownContainer"] {
            color: #000000 !important;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] strong,
        [data-testid="stMarkdownContainer"] span {
            color: #000000 !important;
        }

        /* Input text stays white */
        .stTextArea textarea {
            background: #1e293b !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            caret-color: #ffffff !important;
        }

        .stTextArea textarea::placeholder {
            color: rgba(255,255,255,0.7) !important;
        }

        /* Screenshot */
        div[data-testid="stImage"] img {
            border-radius: 18px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Main
# ============================================================

def main():

    st.set_page_config(
        page_title="Answer Viewer",
        layout="wide",
    )

    setup_styles()

    # ========================================================
    # Session state
    # ========================================================

    if "query" not in st.session_state:
        st.session_state.query = ""

    if "answer" not in st.session_state:
        st.session_state.answer = ""

    if "source" not in st.session_state:
        st.session_state.source = []

    backend_url = DEFAULT_BACKEND_URL

    # ========================================================
    # Clear callback
    # ========================================================

    def clear_query():
        st.session_state.query = ""
        st.session_state.answer = ""
        st.session_state.source = []

    # ========================================================
    # Query
    # ========================================================

    st.subheader("Query")

    st.text_area(
        "Query",
        placeholder="Type your question here...",
        height=120,
        key="query",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)

    search_pressed = col1.button(
        "Search",
        width="stretch",
    )

    col2.button(
        "Clear",
        width="stretch",
        on_click=clear_query,
    )

    # ========================================================
    # Search
    # ========================================================

    if search_pressed:

        query = st.session_state.query.strip()

        if not query:

            st.error(
                "Please enter a query before searching."
            )

        else:

            st.session_state.answer = ""
            st.session_state.source = []

            with st.spinner(
                "Fetching response..."
            ):

                try:

                    data = ask_backend(
                        backend_url,
                        query,
                    )

                    st.session_state.answer = data.get(
                        "answer",
                        "",
                    )

                    st.session_state.source = data.get(
                        "source",
                        [],
                    )

                except requests.RequestException as exc:

                    st.error(
                        f"Backend call failed: {exc}"
                    )

                except ValueError as exc:

                    st.error(
                        f"Invalid response: {exc}"
                    )

    # ========================================================
    # Answer
    # ========================================================

    answer = st.session_state.answer or ""

    if answer:
        render_answer(answer)

    # ========================================================
    # Sources
    # ========================================================

    sources = st.session_state.source or []

    if sources:

        st.header("Metadata")

        for index, source in enumerate(
            sources,
            start=1,
        ):

            render_metadata(
                source,
                index,
            )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()

