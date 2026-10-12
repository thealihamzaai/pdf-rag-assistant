# PDF RAG Assistant

A Flask-based retrieval-augmented generation (RAG) application for asking grounded questions about uploaded PDF documents.

The application extracts text from PDFs, creates semantic embeddings, stores them in a FAISS vector index, retrieves the most relevant passages for each question, and uses a Groq-hosted LLM to generate an answer grounded in the retrieved context.

## Features

- Upload and manage multiple PDF documents
- Page-aware PDF text extraction
- Semantic embeddings with Sentence Transformers
- FAISS similarity search
- Context-grounded LLM answers
- Source document, page, and similarity metadata
- Configurable retrieval and chunking settings
- Local persistence for the FAISS index and metadata
- Clean Flask web interface

## Architecture

```text
PDFs
  ↓
Text extraction
  ↓
Page-aware chunking
  ↓
Sentence Transformer embeddings
  ↓
FAISS vector index
  ↓
User question → query embedding → similarity search
  ↓
Retrieved PDF context
  ↓
Groq LLM
  ↓
Grounded answer + source references
```

## Tech Stack

- Python
- Flask
- Groq API
- Sentence Transformers
- FAISS
- PyPDF2
- NumPy

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the example environment file:

```bash
cp .env.example .env
```

4. Add your Groq API key to `.env`.
5. Start the application:

```bash
python app.py
```

6. Open `http://127.0.0.1:5000` in your browser.

## Configuration

The `.env.example` file documents the available settings, including the Groq model, embedding model, retrieval depth, similarity threshold, chunk size, overlap, and upload size limit.

## Security and Data Notes

- API keys belong in `.env` and must never be committed.
- Uploaded PDFs, generated FAISS indexes, and metadata are excluded from Git by default.
- The application accepts PDF files only and sanitizes filenames.
- Answers are instructed to stay within retrieved document context, but LLM output can still be imperfect and should be verified for important use cases.
- This sample is intended for local development and technical demonstration. Production deployment should add authentication, authorization, rate limiting, hardened file scanning, secure storage, and a production WSGI server.

## Limitations

- Text extraction works best with text-based PDFs. Scanned/image-only documents require OCR, which is not included.
- Character-based chunking is intentionally simple; larger systems may benefit from token-aware or structure-aware chunking.
- Rebuilding the local FAISS index is suitable for small-to-medium demo collections, not large-scale document ingestion.

## Portfolio Context

This project demonstrates a practical end-to-end RAG pipeline: document ingestion, embeddings, vector retrieval, grounded generation, source attribution, and a usable web interface.
