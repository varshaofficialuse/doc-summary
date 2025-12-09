# =================================================================
# main.py — Universal RAG Backend (Full Features)
# Supports ANY file + metadata + delete + list + overlap chunks
# =================================================================

import os
import requests
import fitz  # PDF
import docx
import pytesseract
from PIL import Image
import pandas as pd
from pptx import Presentation
from bs4 import BeautifulSoup
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient, models
from pydantic import BaseModel
from dotenv import load_dotenv
import uuid
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# =================================================================
# 1. Environment
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
# 2. Qdrant setup
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
# 3. Ollama LLM + Embeddings
# =================================================================
def embed_text(text: str):
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}
    )
    return r.json()["embedding"]

def generate_llm(prompt: str):
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt}
    )
    return r.json()["response"]

# =================================================================
# 4. File extractors (based on extension)
# =================================================================
def extract_pdf(path):
    text = ""
    pdf = fitz.open(path)
    for p in pdf:
        text += p.get_text()
    return text

def extract_docx(path):
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)

def extract_txt(path):
    with open(path, "r", errors="ignore") as f:
        return f.read()

def extract_image(path):
    img = Image.open(path)
    return pytesseract.image_to_string(img)

def extract_csv(path):
    return pd.read_csv(path).to_string()

def extract_excel(path):
    return pd.read_excel(path).to_string()

def extract_pptx(path):
    prs = Presentation(path)
    text = ""
    for s in prs.slides:
        for shape in s.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

def extract_html(path):
    with open(path, "r", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        return soup.get_text()

def extract_any(path):
    with open(path, "rb") as f:
        return f.read().decode("latin1", errors="ignore")

def extract_file(path, filename):
    ext = filename.split(".")[-1].lower()

    if ext == "pdf": return extract_pdf(path)
    if ext == "docx": return extract_docx(path)
    if ext == "txt": return extract_txt(path)
    if ext in ["png", "jpg", "jpeg"]: return extract_image(path)
    if ext == "csv": return extract_csv(path)
    if ext == "xlsx": return extract_excel(path)
    if ext == "pptx": return extract_pptx(path)
    if ext in ["html", "htm"]: return extract_html(path)

    return extract_any(path)

# =================================================================
# 5. Smart chunking (sentence-aware + overlap)
# =================================================================

def smart_chunk_text(text, max_tokens=500, overlap=50):
    sentences = sent_tokenize(text)
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) < max_tokens:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current:
        chunks.append(current.strip())

    # Add overlap
    final_chunks = []
    for i in range(len(chunks)):
        start = max(0, i - 1)
        combined = " ".join(chunks[start:i+1])
        final_chunks.append(combined)

    return final_chunks

# =================================================================
# 6. Upload chunks with metadata
# =================================================================

def store_chunk(text, filename):
    emb = embed_text(text)
    doc_id = str(uuid.uuid4())  # unique ID per chunk

    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=doc_id,
                vector=emb,
                payload={
                    "text": text,
                    "filename": filename
                }
            )
        ]
    )

# =================================================================
# 7. Search
# =================================================================

def search_docs(q_emb, k=5):
    return qdrant.search(
        collection_name=COLLECTION,
        query_vector=q_emb,
        limit=k
    )

# =================================================================
# FASTAPI
# =================================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================================================================
# 1. Upload MULTIPLE files
# =================================================================

@app.post("/upload-files")
async def upload_multiple_files(files: list[UploadFile] = File(...)):
    file_summaries = []

    for file in files:
        temp = f"temp_{file.filename}"
        with open(temp, "wb") as f:
            f.write(await file.read())

        extracted = extract_file(temp, file.filename)
        chunks = smart_chunk_text(extracted)

        for c in chunks:
            store_chunk(c, file.filename)

        file_summaries.append({
            "filename": file.filename,
            "chunks_stored": len(chunks)
        })

    return {"uploaded": file_summaries}

# =================================================================
# 2. View uploaded files
# =================================================================

@app.get("/list-files")
def list_files():
    points = qdrant.scroll(collection_name=COLLECTION, limit=100000)[0]
    filenames = sorted(list({p.payload["filename"] for p in points}))
    return {"files": filenames}

# =================================================================
# 3. Delete document by filename
# =================================================================

class DeleteReq(BaseModel):
    filename: str

@app.post("/delete-file")
def delete_file(req: DeleteReq):
    qdrant.delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(value=req.filename)
                    )
                ]
            )
        )
    )
    return {"status": "deleted", "filename": req.filename}

# =================================================================
# 4. Ask Questions
# =================================================================

class AskReq(BaseModel):
    query: str

@app.post("/ask")
def ask_question(req: AskReq):
    q_emb = embed_text(req.query)
    hits = search_docs(q_emb)

    context = "\n".join([h.payload["text"] for h in hits])

    prompt = f"""
    Use ONLY the context below to answer the question.
    CONTEXT:
    {context}

    QUESTION:
    {req.query}
    """

    answer = generate_llm(prompt)
    return {"answer": answer, "context_used": context}
