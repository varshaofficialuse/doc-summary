# RAG GenAI Backend (Production-Ready POC)

This is a production-style backend built with **Python + FastAPI + OpenAI + FAISS**.

## Features

- Upload documents in multiple formats: **PDF, DOCX, TXT, MD, PNG, JPG, JPEG, WEBP**
- Extract text (including OCR for images using Tesseract)
- Generate a **summary** of the uploaded content using LLM
- Build a **RAG index** over the document and provide a **chat endpoint**
- For image uploads, support **image transformation** with a text prompt using OpenAI image edit API
- Proper error handling, logging, validation, and clean JSON responses

## Project structure

- `main.py` – FastAPI app with all endpoints and logic
- `requirements.txt` – Python dependencies
- `data/` – Created at runtime; contains uploads, FAISS indexes, and metadata

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install **Tesseract OCR** on your system (required for image text extraction):

- Ubuntu/Debian: `sudo apt install tesseract-ocr`
- macOS (with Homebrew): `brew install tesseract`
- Windows: download from the official Tesseract site and add it to PATH.

4. Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."        # Linux / macOS
set OPENAI_API_KEY=sk-...             # Windows (CMD)
$env:OPENAI_API_KEY="sk-..."          # Windows (PowerShell)
```

5. Run the server:

```bash
uvicorn main:app --reload
```

## Endpoints

### `POST /documents/upload`

Upload a file and get:

- `document_id`
- original filename
- auto-generated summary
- document type (`text` or `image`)

Example (curl):

```bash
curl -X POST "http://localhost:8000/documents/upload"   -F "file=@/path/to/your/file.pdf"
```

### `POST /chat/{document_id}`

Ask questions about a previously uploaded document (RAG).

Body:

```json
{
  "question": "What is the main topic?"
}
```

### `POST /images/{document_id}/transform`

For image documents, apply an AI-based transformation using a prompt.

Body:

```json
{
  "prompt": "Make this look like a pencil sketch with high contrast"
}
```

Response contains `image_b64`, a base64-encoded PNG. On frontend you can render it as:

```js
<img src={"data:image/png;base64," + image_b64} />
```

### `GET /health`

Simple health check.

---

This backend is suitable as a **GenAI + RAG POC** and can be deployed to platforms like Render, Railway, or any VM with Python 3.10+.
