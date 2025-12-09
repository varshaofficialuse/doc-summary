# =================================================================
# main.py — Universal RAG Backend
# Supports: PDF, DOCX, TXT, PPTX, XLSX, CSV, Images, HTML, MD, ALL FILES
# Chunk Upload + Large File Handling + FastAPI + Qdrant + Ollama
# =================================================================

import os
import requests
import fitz  # PyMuPDF for PDF
import docx
import pytesseract
from PIL import Image
import pandas as pd
from pptx import Presentation
import magic
from bs4 import BeautifulSoup
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from pydantic import BaseModel

# =================================================================
# 1. Environment setup
# =================================================================
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION = os.getenv("VECTOR_COLLECTION_NAME", "documents")

# =================================================================
# 2. Qdrant connection (local + cloud)
# =================================================================

if QDRANT_URL:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def ensure_collection():
    try:
        qdrant.get_collection(COLLECTION)
    except:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
        )

ensure_collection()

# =================================================================
# 3. Ollama—Embedding + LLM
# =================================================================

def embed_text(text: str):
    """Generate embedding using Ollama."""
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}
    )
    return r.json()["embedding"]

def generate_llm(prompt: str):
    """Generate answer using Ollama."""
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt},
        stream=False
    )
    return r.json()["response"]

# =================================================================
# 4. File extractors — universal support
# =================================================================

def extract_pdf(path):
    text = ""
    pdf = fitz.open(path)
    for page in pdf:
        text += page.get_text()
    return text

def extract_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_txt(path):
    with open(path, "r", errors="ignore") as f:
        return f.read()

def extract_image(path):
    img = Image.open(path)
    return pytesseract.image_to_string(img)

def extract_csv(path):
    df = pd.read_csv(path)
    return df.to_string()

def extract_excel(path):
    df = pd.read_excel(path)
    return df.to_string()

def extract_pptx(path):
    prs = Presentation(path)
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

def extract_html(path):
    with open(path, "r", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        return soup.get_text()

def extract_any(path):
    """Fallback extractor: read as plain text."""
    with open(path, "rb") as f:
        return f.read().decode("latin1", errors="ignore")

# =================================================================
# 5. Auto-detect and extract ANY file
# =================================================================

def extract_file_content(path):
    file_type = magic.from_file(path, mime=True)

    if "pdf" in file_type:
        return extract_pdf(path)
    if "word" in file_type:
        return extract_docx(path)
    if "text" in file_type:
        return extract_txt(path)
    if "image" in file_type:
        return extract_image(path)
    if "csv" in file_type:
        return extract_csv(path)
    if "excel" in file_type:
        return extract_excel(path)
    if "presentation" in file_type:
        return extract_pptx(path)
    if "html" in file_type or "xml" in file_type:
        return extract_html(path)

    # fallback
    return extract_any(path)

# =================================================================
# 6. Chunk generator (supports ANY SIZE FILE)
# =================================================================

def chunk_text(text, max_chars=1500):
    """Chunk big documents efficiently."""
    for i in range(0, len(text), max_chars):
        yield text[i : i + max_chars]

# =================================================================
# 7. Store documents into Qdrant
# =================================================================

def store_chunk(doc_id, text):
    emb = embed_text(text)
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=doc_id,
                vector=emb,
                payload={"text": text}
            )
        ]
    )

# =================================================================
# 8. Search with Qdrant
# =================================================================

def search_docs(q_emb, k=5):
    return qdrant.search(
        collection_name=COLLECTION,
        query_vector=q_emb,
        limit=k
    )

# =================================================================
# 9. FastAPI server
# =================================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================================================================
# Upload file — supports ALL formats
# =================================================================

@app.post("/upload-file")
async def upload_any_file(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    extracted = extract_file_content(temp_path)

    # Chunk + upload
    doc_id = 1
    for chunk in chunk_text(extracted):
        store_chunk(doc_id, chunk)
        doc_id += 1

    return {"status": "uploaded", "chunks": doc_id - 1}

# =================================================================
# Ask questions API
# =================================================================

class Ask(BaseModel):
    query: str

@app.post("/ask")
def ask(req: Ask):
    q_emb = embed_text(req.query)
    hits = search_docs(q_emb)
    context = "\n".join([h.payload["text"] for h in hits])

    prompt = f"""
    Answer the question using ONLY the context below.
    CONTEXT:
    {context}

    QUESTION:
    {req.query}
    """

    answer = generate_llm(prompt)
    return {"answer": answer, "context": context}
