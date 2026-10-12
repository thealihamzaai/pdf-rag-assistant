import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for
from groq import Groq
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "pdfs"
INDEX_FILE = BASE_DIR / "faiss_index.bin"
METADATA_FILE = BASE_DIR / "metadata.json"
UPLOAD_FOLDER.mkdir(exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
TOP_K = int(os.getenv("TOP_K", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
embedding_dimension = embedding_model.get_sentence_embedding_dimension()
client = Groq(api_key=GROQ_API_KEY)


def new_index():
    return faiss.IndexFlatIP(embedding_dimension)


def load_index():
    if INDEX_FILE.exists() and METADATA_FILE.exists():
        try:
            loaded_index = faiss.read_index(str(INDEX_FILE))
            loaded_metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
            if loaded_index.d == embedding_dimension and loaded_index.ntotal == len(loaded_metadata):
                return loaded_index, loaded_metadata
        except Exception:
            pass
    return new_index(), []


index, metadata = load_index()


def save_index():
    faiss.write_index(index, str(INDEX_FILE))
    METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_pages(pdf_path: Path):
    pages = []
    reader = PdfReader(str(pdf_path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((page_number, text))
    return pages


def chunk_text(text: str):
    chunks = []
    step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
    for start in range(0, len(text), step):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        if start + CHUNK_SIZE >= len(text):
            break
    return chunks


def create_embeddings(texts):
    vectors = embedding_model.encode(texts, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors


def rebuild_index():
    global index, metadata
    index = new_index()
    metadata = []

    for pdf_path in sorted(UPLOAD_FOLDER.glob("*.pdf")):
        try:
            for page_number, page_text in extract_pages(pdf_path):
                chunks = chunk_text(page_text)
                if not chunks:
                    continue
                vectors = create_embeddings(chunks)
                start_id = len(metadata)
                index.add(vectors)
                for offset, chunk in enumerate(chunks):
                    metadata.append(
                        {
                            "chunk_id": start_id + offset,
                            "source": pdf_path.name,
                            "page": page_number,
                            "text": chunk,
                        }
                    )
        except Exception as exc:
            app.logger.warning("Skipped %s: %s", pdf_path.name, exc)

    save_index()


def retrieve_chunks(question: str):
    if index.ntotal == 0:
        return []

    query_vector = create_embeddings([question])
    search_k = min(TOP_K, index.ntotal)
    scores, indices = index.search(query_vector, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        similarity = float(score)
        if similarity < SIMILARITY_THRESHOLD:
            continue
        item = metadata[idx]
        results.append(
            {
                "chunk_id": item["chunk_id"],
                "source": item["source"],
                "page": item["page"],
                "text": item["text"],
                "score": round(similarity * 100, 1),
            }
        )
    return results


def generate_answer(question: str, chunks):
    if not chunks:
        return "I couldn't find enough relevant information in the uploaded PDFs to answer that confidently."

    context_blocks = []
    for chunk in chunks:
        context_blocks.append(
            f"Source: {chunk['source']} | Page: {chunk['page']}\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
You are a retrieval-augmented assistant answering questions only from the supplied PDF context.

Rules:
- Use only the supplied context for factual claims.
- If the context is insufficient, say so clearly.
- Do not invent facts, quotations, page numbers, or sources.
- Keep the answer clear and concise unless the question needs detail.
- When useful, cite the supporting document and page in parentheses, e.g. (report.pdf, p. 3).

Context:
{context}

Question:
{question}
""".strip()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": "You are a careful document-grounded RAG assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def list_pdfs():
    return sorted(path.name for path in UPLOAD_FOLDER.glob("*.pdf"))


@app.get("/")
def home():
    return render_template("index.html", files=list_pdfs(), max_upload_mb=MAX_UPLOAD_MB)


@app.post("/upload")
def upload():
    files = request.files.getlist("files")
    saved_any = False

    for file in files:
        if not file or not file.filename:
            continue
        filename = secure_filename(file.filename)
        if not filename.lower().endswith(".pdf"):
            continue
        file.save(UPLOAD_FOLDER / filename)
        saved_any = True

    if saved_any:
        rebuild_index()
    return redirect(url_for("home"))


@app.post("/delete/<path:filename>")
def delete_pdf(filename):
    safe_name = secure_filename(filename)
    if safe_name != filename or not safe_name.lower().endswith(".pdf"):
        return jsonify({"error": "Invalid filename."}), 400

    pdf_path = UPLOAD_FOLDER / safe_name
    if pdf_path.exists():
        pdf_path.unlink()
        rebuild_index()
    return redirect(url_for("home"))


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("message", "")).strip()
    if not question:
        return jsonify({"answer": "Please enter a question.", "sources": []}), 400

    try:
        chunks = retrieve_chunks(question)
        answer = generate_answer(question, chunks)
        public_sources = [
            {
                "source": item["source"],
                "page": item["page"],
                "chunk_id": item["chunk_id"],
                "score": item["score"],
            }
            for item in chunks
        ]
        return jsonify({"answer": answer, "sources": public_sources})
    except Exception as exc:
        app.logger.exception("Chat request failed")
        return jsonify({"answer": "The request could not be completed. Please try again.", "sources": []}), 500


@app.errorhandler(413)
def too_large(_error):
    return f"Upload too large. Maximum request size is {MAX_UPLOAD_MB} MB.", 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
