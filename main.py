
import os
import uuid
import json
import logging
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import faiss
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

from openai import OpenAI

from pypdf import PdfReader
import docx
from PIL import Image, UnidentifiedImageError
import pytesseract

# =========================
# Logging setup
# =========================

logger = logging.getLogger("rag_genai_backend")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# =========================
# Config & setup
# =========================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set in environment")
    raise RuntimeError("OPENAI_API_KEY not set in environment")

try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    logger.exception("Failed to initialize OpenAI client")
    raise

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "indexes"
META_DIR = DATA_DIR / "meta"

for d in [DATA_DIR, UPLOAD_DIR, INDEX_DIR, META_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# limits
MAX_FILE_SIZE_MB = 25
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp"}


# =========================
# Pydantic models
# =========================

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    summary: str
    type: Literal["text", "image"]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class ChatResponse(BaseModel):
    answer: str


class ImageTransformRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=800)


class ImageTransformResponse(BaseModel):
    image_b64: str  # base64-encoded PNG


# =========================
# Utility: file saving
# =========================

def _validate_extension(path: Path) -> None:
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning("Attempt to upload unsupported extension: %s", ext)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. "
                   f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )


def _validate_size(path: Path) -> None:
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        logger.warning("File too large: %.2f MB", size_mb)
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.2f} MB). Max allowed is {MAX_FILE_SIZE_MB} MB.",
        )


def save_upload(file: UploadFile) -> Tuple[Path, str]:
    """Save the uploaded file to disk and return (path, document_id)."""
    ext = Path(file.filename).suffix or ""
    doc_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{doc_id}{ext}"

    try:
        with dest.open("wb") as f:
            # stream to avoid holding in memory
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        logger.exception("Error while saving uploaded file")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file") from e
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    _validate_extension(dest)
    _validate_size(dest)

    logger.info("Saved upload %s as %s", file.filename, dest)
    return dest, doc_id


# =========================
# Text extraction
# =========================

def extract_text_from_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
    except Exception as e:
        logger.exception("Failed to read PDF: %s", path)
        raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file") from e

    text_parts = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text.strip():
            text_parts.append(page_text)

    text = "\n".join(text_parts).strip()
    if not text:
        logger.warning("No text extracted from PDF: %s", path)
    return text


def extract_text_from_docx(path: Path) -> str:
    try:
        doc = docx.Document(str(path))
    except Exception as e:
        logger.exception("Failed to read DOCX: %s", path)
        raise HTTPException(status_code=400, detail="Invalid or corrupted DOCX file") from e

    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs).strip()
    if not text:
        logger.warning("No text extracted from DOCX: %s", path)
    return text


def extract_text_from_txt(path: Path) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            text = path.read_text(encoding=enc)
            if text.strip():
                return text
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.exception("Error reading text file: %s", path)
            raise HTTPException(status_code=400, detail="Failed to read text file") from e

    logger.warning("No readable text from TXT: %s", path)
    return ""


def extract_text_from_image(path: Path) -> str:
    try:
        img = Image.open(path)
    except UnidentifiedImageError:
        logger.exception("Unidentified image format: %s", path)
        raise HTTPException(status_code=400, detail="Unsupported or corrupted image format")
    except Exception as e:
        logger.exception("Failed to open image: %s", path)
        raise HTTPException(status_code=400, detail="Failed to open image") from e

    try:
        text = pytesseract.image_to_string(img)
    except Exception as e:
        logger.exception("Tesseract OCR failed")
        raise HTTPException(
            status_code=500,
            detail="OCR engine failed while reading image. Ensure Tesseract is installed on the server.",
        ) from e

    text = (text or "").strip()
    if not text:
        logger.info("Image OCR produced no text for %s", path)
        return "No readable text found in the image."
    return text


def extract_text(path: Path, content_type: str) -> Tuple[str, str]:
    """
    Returns (text, doc_type) where doc_type is 'text' or 'image'.
    """
    suffix = path.suffix.lower()

    # Image
    if content_type.startswith("image/") or suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return extract_text_from_image(path), "image"

    # PDF
    if suffix == ".pdf" or content_type == "application/pdf":
        return extract_text_from_pdf(path), "text"

    # DOCX
    if suffix == ".docx" or "word" in (content_type or "").lower():
        return extract_text_from_docx(path), "text"

    # Plain text
    if suffix in {".txt", ".md"} or (content_type or "").startswith("text/"):
        return extract_text_from_txt(path), "text"

    logger.warning("Unsupported file type detected in extract_text: %s (%s)", suffix, content_type)
    raise HTTPException(status_code=400, detail="Unsupported file type")


# =========================
# Chunking
# =========================

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == length:
            break
        start += chunk_size - overlap

    return chunks


# =========================
# OpenAI helpers
# =========================

def get_embedding(text: str) -> List[float]:
    try:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return resp.data[0].embedding
    except Exception as e:
        logger.exception("Failed to get embedding from OpenAI")
        raise HTTPException(status_code=502, detail="Embedding service unavailable") from e


def summarize_text(text: str, is_image: bool = False) -> str:
    if not text.strip():
        return "No meaningful text was found to summarize."

    base_prompt = (
        "You are an AI summarizer. "
        "Summarize the following document in 3–6 concise bullet points, "
        "focusing on key topics, entities, and any important numbers.\n\n"
        "Document content:\n"
    )
    if is_image:
        base_prompt = (
            "You are summarizing text extracted from an image (like a screenshot or scanned page). "
            "Summarize in 3–6 bullet points, mention that it came from an image if relevant, "
            "and be explicit about any visible structure or key sections.\n\n"
            "Extracted content:\n"
        )

    prompt = base_prompt + text[:8000]

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.exception("Failed to summarize text")
        raise HTTPException(status_code=502, detail="LLM summarization service unavailable") from e


def rag_answer(question: str, chunks: List[str], index: faiss.IndexFlatL2) -> str:
    if not chunks:
        return "I don't have any content for this document yet."

    try:
        q_emb = np.array([get_embedding(question)], dtype="float32")
        k = min(5, len(chunks))
        distances, indices = index.search(q_emb, k)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("FAISS search failed")
        raise HTTPException(status_code=500, detail="Internal search error") from e

    top_chunks = [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
    if not top_chunks:
        return "I could not find relevant parts in the document for your question."

    context = "\n\n---\n\n".join(top_chunks)
    prompt = (
        "You are a helpful assistant that answers questions about a single uploaded document.\n"
        "Use ONLY the information in the provided context. "
        "If the answer is not present in the context, explicitly say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly and concisely:"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.exception("Failed to generate RAG answer")
        raise HTTPException(status_code=502, detail="LLM question-answering service unavailable") from e


def edit_image_with_prompt(path: Path, prompt: str) -> str:
    """
    Use OpenAI image edit API to create a modified version
    of the uploaded image according to the user's prompt.
    Returns base64-encoded PNG string.
    """
    try:
        with path.open("rb") as img_file:
            # New OpenAI Images API: send raw bytes for editing
            img_bytes = img_file.read()

        result = client.images.edit(
            model="gpt-image-1",
            image=img_bytes,
            prompt=prompt,
            size="1024x1024",
            response_format="b64_json",
        )
        b64 = result.data[0].b64_json
        return b64
    except Exception as e:
        logger.exception("Image transformation failed")
        raise HTTPException(status_code=502, detail="Image transformation service unavailable") from e


# =========================
# FAISS persistence
# =========================

def build_and_save_index(doc_id: str, chunks: List[str]) -> None:
    if not chunks:
        logger.warning("Attempted to build FAISS index with no chunks: %s", doc_id)
        raise HTTPException(status_code=400, detail="No content available to index")

    try:
        embeddings = [get_embedding(c) for c in chunks]
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        vectors = np.array(embeddings, dtype="float32")
        index.add(vectors)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to build FAISS index")
        raise HTTPException(status_code=500, detail="Failed to build search index") from e

    index_path = INDEX_DIR / f"{doc_id}.index"
    chunks_path = INDEX_DIR / f"{doc_id}_chunks.json"

    try:
        faiss.write_index(index, str(index_path))
        chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.exception("Failed to persist FAISS index or chunks")
        raise HTTPException(status_code=500, detail="Failed to persist document index") from e

    logger.info("Index built and saved for document %s", doc_id)


def load_index_and_chunks(doc_id: str):
    index_path = INDEX_DIR / f"{doc_id}.index"
    chunks_path = INDEX_DIR / f"{doc_id}_chunks.json"

    if not index_path.exists() or not chunks_path.exists():
        logger.warning("Index or chunks missing for document %s", doc_id)
        raise HTTPException(status_code=404, detail="Index not found for this document")

    try:
        index = faiss.read_index(str(index_path))
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.exception("Failed to load index or chunks for %s", doc_id)
        raise HTTPException(status_code=500, detail="Failed to load search index") from e

    return index, chunks


# =========================
# Metadata persistence
# =========================

def save_meta(doc_id: str, meta: dict) -> None:
    meta_path = META_DIR / f"{doc_id}.json"
    try:
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.exception("Failed to save metadata for %s", doc_id)
        raise HTTPException(status_code=500, detail="Failed to persist document metadata") from e


def load_meta(doc_id: str) -> dict:
    meta_path = META_DIR / f"{doc_id}.json"
    if not meta_path.exists():
        logger.warning("Metadata not found for document %s", doc_id)
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.exception("Failed to read metadata for %s", doc_id)
        raise HTTPException(status_code=500, detail="Failed to read document metadata") from e


# =========================
# FastAPI app
# =========================

app = FastAPI(title="RAG GenAI POC Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handlers for cleaner error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning("HTTPException: %s %s -> %s", request.method, request.url.path, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception for %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    1. Save file
    2. Extract text
    3. Summarize
    4. Chunk + embed + build FAISS index
    5. Save metadata
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    logger.info("Received upload: %s (%s)", file.filename, file.content_type)

    saved_path, doc_id = save_upload(file)

    try:
        text, doc_type = extract_text(saved_path, file.content_type or "")
    except HTTPException:
        # propagate structured errors
        raise
    except Exception as e:
        logger.exception("Unexpected error during text extraction")
        raise HTTPException(status_code=500, detail="Unexpected error while extracting text") from e

    if not text.strip():
        logger.warning("No text extracted for document %s", doc_id)
        raise HTTPException(status_code=400, detail="No text could be extracted from the document")

    summary = summarize_text(text, is_image=(doc_type == "image"))
    chunks = chunk_text(text)

    if not chunks:
        logger.warning("No chunks generated for document %s", doc_id)
        raise HTTPException(status_code=400, detail="Unable to create content chunks from document")

    build_and_save_index(doc_id, chunks)

    meta = {
        "document_id": doc_id,
        "filename": file.filename,
        "stored_path": str(saved_path),
        "type": doc_type,
        "summary": summary,
    }
    save_meta(doc_id, meta)

    logger.info("Document %s processed successfully", doc_id)

    return UploadResponse(
        document_id=doc_id,
        filename=file.filename,
        summary=summary,
        type=doc_type,
    )


@app.post("/chat/{document_id}", response_model=ChatResponse)
async def chat_with_document(document_id: str, payload: ChatRequest):
    """
    RAG-style Q&A over a single uploaded document.
    """
    logger.info("Chat request for document %s", document_id)

    try:
        meta = load_meta(document_id)
        index, chunks = load_index_and_chunks(document_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error loading data for chat with document %s", document_id)
        raise HTTPException(status_code=500, detail="Failed to prepare document for Q&A") from e

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    answer = rag_answer(question, chunks, index)

    return ChatResponse(answer=answer)


@app.post("/images/{document_id}/transform", response_model=ImageTransformResponse)
async def transform_image(document_id: str, payload: ImageTransformRequest):
    """
    Image modification/animation-style effect based on user prompt.
    Uses original uploaded image of this document.
    """
    logger.info("Image transform request for document %s", document_id)

    meta = load_meta(document_id)
    if meta.get("type") != "image":
        raise HTTPException(status_code=400, detail="This document is not an image")

    img_path = Path(meta["stored_path"])
    if not img_path.exists():
        logger.error("Original image file missing for document %s", document_id)
        raise HTTPException(status_code=404, detail="Original image file not found")

    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    image_b64 = edit_image_with_prompt(img_path, prompt)

    return ImageTransformResponse(image_b64=image_b64)


@app.get("/health")
async def health():
    return {"status": "ok"}


# For local dev:
#   uvicorn main:app --reload
