"""Smart Document Knowledge Assistant - a small RAG app built with Streamlit.

Pipeline:  PDF/TXT -> text -> chunks -> embeddings -> FAISS -> top-k chunks -> Gemini -> answer + sources
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------- settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks
TOP_K = 5             # chunks retrieved per question
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"  # used if GEMINI_MODEL is not set in .env
SAMPLE_FILE = Path(__file__).parent / "data" / "sample_notes.txt"
NOT_FOUND_MESSAGE = "I couldn't find this information in the uploaded documents."


# ---------------------------------------------------------------- 1. extraction
def extract_text(filename: str, data: bytes) -> list[tuple[int | None, str]]:
    """Return a list of (page_number, text). TXT files have page_number None.

    Raises ValueError with a friendly message for unsupported, broken or empty files.
    """
    name = filename.lower()
    pages: list[tuple[int | None, str]] = []

    if name.endswith(".txt"):
        text = data.decode("utf-8", errors="ignore").strip()
        if text:
            pages.append((None, text))
    elif name.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            for number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append((number, text))
        except Exception as error:
            raise ValueError(f"Could not read PDF '{filename}': {error}")
    else:
        raise ValueError(f"'{filename}' is not supported. Please upload PDF or TXT files.")

    if not pages:
        raise ValueError(
            f"'{filename}' contains no extractable text (it may be empty or a scanned image PDF)."
        )
    return pages


# ---------------------------------------------------------------- 2. chunking
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping pieces.

    Overlap keeps a sentence that falls on a boundary from losing its context.
    """
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += size - overlap
    return chunks


def build_chunks(docs: dict) -> list[dict]:
    """Turn all documents into chunks, each carrying its metadata."""
    chunks = []
    for filename, doc in docs.items():
        number = 0
        for page, text in doc["pages"]:
            for piece in chunk_text(text):
                number += 1
                chunks.append({"filename": filename, "page": page, "chunk": number, "text": piece})
    return chunks


# ---------------------------------------------------------------- 3. embeddings
@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)...")
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Turn text into vectors. Similar meaning -> vectors that point the same way."""
    vectors = load_embedder().encode(texts, convert_to_numpy=True, show_progress_bar=False)
    vectors = vectors.astype("float32")
    faiss.normalize_L2(vectors)  # unit length, so inner product == cosine similarity
    return vectors


# ---------------------------------------------------------------- 4. FAISS
def build_index(chunks: list[dict]) -> faiss.Index:
    """Store all chunk vectors in a FAISS index for fast similarity search."""
    vectors = embed_texts([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def search(query: str, index: faiss.Index, chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Embed the question and return the top_k most similar chunks."""
    scores, ids = index.search(embed_texts([query]), min(top_k, len(chunks)))
    results = []
    for score, i in zip(scores[0], ids[0]):
        if i != -1:
            results.append({**chunks[i], "score": float(score)})
    return results


# ---------------------------------------------------------------- 5. Gemini
def generate_answer(question: str, retrieved: list[dict], history: list[dict]) -> str:
    """Ask Gemini to answer using ONLY the retrieved chunks."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip()
    if not model or model == "your_model_here":
        model = DEFAULT_GEMINI_MODEL

    context = "\n\n".join(
        f"[Source {n}: {c['filename']}, {location_label(c)}]\n{c['text']}"
        for n, c in enumerate(retrieved, start=1)
    )
    # Recent turns let Gemini understand follow-up questions like "give an example".
    recent = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:])

    system_instruction = (
        "You are a document question-answering assistant. "
        "Answer ONLY using the document context provided. Do not use outside knowledge "
        "and do not invent information. "
        f"If the answer is not in the context, reply exactly: \"{NOT_FOUND_MESSAGE}\" "
        "Otherwise give a clear, concise answer."
    )
    prompt = (
        f"Document context:\n{context}\n\n"
        f"Conversation so far:\n{recent or '(none)'}\n\n"
        f"Question: {question}"
    )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.2),
    )
    return (response.text or "").strip() or NOT_FOUND_MESSAGE


# ---------------------------------------------------------------- 6. sources UI
def location_label(chunk: dict) -> str:
    if chunk["page"] is not None:
        return f"page {chunk['page']}, chunk {chunk['chunk']}"
    return f"chunk {chunk['chunk']}"


def show_sources(sources: list[dict]) -> None:
    st.markdown("**Sources**")
    for n, c in enumerate(sources, start=1):
        label = f"{n}. {c['filename']} - {location_label(c)} (similarity {c['score']:.2f})"
        with st.expander(label):
            st.write(c["text"])


# ---------------------------------------------------------------- app state
def init_state() -> None:
    st.session_state.setdefault("docs", {})        # filename -> {"pages": [...], "origin": ...}
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("index", None)
    st.session_state.setdefault("indexed", ())     # which documents the current index was built from
    st.session_state.setdefault("messages", [])    # chat history
    st.session_state.setdefault("uploader_key", 0)  # changing it resets the file uploader


def rebuild_index() -> None:
    """Build the FAISS index for the loaded documents.

    The index is marked as "built" only if it succeeds, so a failure (for example the embedding
    model could not be downloaded) is shown to the user and retried on the next interaction.
    """
    docs = st.session_state.docs
    st.session_state.chunks = build_chunks(docs)
    st.session_state.index = None
    if st.session_state.chunks:
        try:
            with st.spinner("Creating embeddings and FAISS index (first run downloads the model)..."):
                st.session_state.index = build_index(st.session_state.chunks)
        except Exception as error:
            st.sidebar.error(
                f"Could not create embeddings: {error}\n\n"
                "The embedding model is downloaded from huggingface.co on first use, "
                "so check your internet connection and that all packages installed correctly."
            )
            return
    st.session_state.indexed = tuple(sorted(docs))


def sync_uploaded_files(uploaded_files) -> None:
    """Keep st.session_state.docs in step with the files shown in the uploader."""
    docs = st.session_state.docs
    current = {f.name for f in uploaded_files}

    for name in [n for n, d in docs.items() if d["origin"] == "upload" and n not in current]:
        del docs[name]

    for f in uploaded_files:
        if f.name in docs:
            continue
        try:
            docs[f.name] = {"pages": extract_text(f.name, f.getvalue()), "origin": "upload"}
        except ValueError as error:
            st.sidebar.error(str(error))

    # (Re)build whenever the index does not match the loaded documents.
    if tuple(sorted(docs)) != st.session_state.indexed:
        rebuild_index()


def reset_all() -> None:
    st.session_state.docs = {}
    st.session_state.chunks = []
    st.session_state.index = None
    st.session_state.indexed = ()
    st.session_state.messages = []
    st.session_state.uploader_key += 1


# ---------------------------------------------------------------- main UI
def main() -> None:
    st.set_page_config(page_title="Smart Document Knowledge Assistant", page_icon="📚")
    # override=True so a key added to .env later is picked up on the next interaction
    load_dotenv(override=True)
    init_state()

    st.title("📚 Smart Document Knowledge Assistant")
    st.caption("Upload PDF or TXT files and ask questions. Answers come only from your documents.")

    # ---- sidebar: upload + document list
    with st.sidebar:
        st.header("Documents")
        uploaded = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )
        if st.button("Use sample notes", disabled=not SAMPLE_FILE.exists()):
            try:
                st.session_state.docs[SAMPLE_FILE.name] = {
                    "pages": extract_text(SAMPLE_FILE.name, SAMPLE_FILE.read_bytes()),
                    "origin": "sample",
                }
            except ValueError as error:
                st.error(str(error))
        sync_uploaded_files(uploaded or [])  # also builds the index

        st.subheader("Loaded documents")
        if st.session_state.docs:
            for name in st.session_state.docs:
                st.write(f"📄 {name}")
            if st.session_state.index is not None:
                st.caption(f"{len(st.session_state.chunks)} chunks indexed - ready for questions")
        else:
            st.info("No documents loaded yet.")

        if st.button("Clear / reset"):
            reset_all()
            st.rerun()

        if not os.getenv("GEMINI_API_KEY", "").strip():
            st.warning("No Gemini API key found. Add GEMINI_API_KEY to your .env file to get answers.")

    # ---- chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                show_sources(message["sources"])

    # ---- new question
    question = st.chat_input("Ask a question about your documents")
    if not question:
        return

    with st.chat_message("user"):
        st.write(question)

    if not st.session_state.docs:
        st.warning("Please upload at least one PDF or TXT file (or click 'Use sample notes') first.")
        return
    if st.session_state.index is None:
        st.warning("Your documents are loaded but could not be indexed. See the error in the sidebar.")
        return
    if not os.getenv("GEMINI_API_KEY", "").strip():
        st.warning("Gemini API key is missing. Copy .env.example to .env, add your key, then ask again.")
        return

    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})

    # Short follow-ups ("what about regression?") search better with the previous question added.
    search_query = question
    if len(question.split()) < 6:
        previous = [m["content"] for m in history if m["role"] == "user"]
        if previous:
            search_query = f"{previous[-1]} {question}"

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and asking Gemini..."):
            try:
                retrieved = search(search_query, st.session_state.index, st.session_state.chunks)
                answer = generate_answer(question, retrieved, history)
            except Exception as error:
                st.error(f"Could not get an answer (search or Gemini API error): {error}")
                st.session_state.messages.pop()  # drop the unanswered question
                return
        st.write(answer)
        show_sources(retrieved)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": retrieved})


if __name__ == "__main__":
    main()
