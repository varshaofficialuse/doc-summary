# ============================================================
#  main.py — Full RAG Backend (Single File Version)
#  Uses: FastAPI + Ollama + Qdrant
#  100% Local, Free, No Docker Needed
# ============================================================

import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

# ============================================================
# 1. LOAD ENV VARIABLES
# ============================================================

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION = os.getenv("VECTOR_COLLECTION_NAME", "documents")

# ============================================================
# 2. QDRANT CLIENT + COLLECTION SETUP
# ============================================================

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def ensure_collection():
    """Create the vector collection if it doesn't exist."""
    try:
        qdrant.get_collection(COLLECTION)
        print("Qdrant collection already exists:", COLLECTION)
    except:
        print("Creating Qdrant collection:", COLLECTION)
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=768, distance=models.Distance.COSINE
            )
        )

ensure_collection()

# ============================================================
# 3. OLLAMA FUNCTIONS (LLM + Embeddings)
# ============================================================

def embed_text(text: str):
    """Generate embedding using Ollama."""
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}
    )
    return r.json()["embedding"]

def generate_llm(prompt: str) -> str:
    """Generate text using Ollama LLM."""
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt},
        stream=False
    )
    return r.json()["response"]

# ============================================================
# 4. QDRANT STORAGE + SEARCH FUNCTIONS
# ============================================================

def store_document(doc_id: int, text: str, embedding):
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=doc_id,
                vector=embedding,
                payload={"text": text}
            )
        ]
    )

def search_docs(query_embedding, k=3):
    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_embedding,
        limit=k
    )
    return results

# ============================================================
# 5. RAG PIPELINE: Add Document + Ask Question
# ============================================================

def rag_add_document(doc_id: int, text: str):
    embedding = embed_text(text)
    store_document(doc_id, text, embedding)
    return {"status": "stored", "id": doc_id}

def rag_ask_question(query: str):
    q_embed = embed_text(query)

    hits = search_docs(q_embed, k=5)
    context = "\n".join([hit.payload["text"] for hit in hits])

    prompt = f"""
    Use ONLY the provided context to answer the question.

    CONTEXT:
    {context}

    QUESTION: {query}
    """

    answer = generate_llm(prompt)
    return {"answer": answer, "context": context}

# ============================================================
# 6. FASTAPI MODELS
# ============================================================

class AddDocumentRequest(BaseModel):
    doc_id: int
    text: str

class AskQuery(BaseModel):
    query: str

# ============================================================
# 7. FASTAPI ROUTES
# ============================================================

app = FastAPI()

@app.post("/add")
def add_doc_api(req: AddDocumentRequest):
    return rag_add_document(req.doc_id, req.text)

@app.post("/ask")
def ask_api(req: AskQuery):
    return rag_ask_question(req.query)

# ============================================================
# 8. RUN SERVER (manual)
# ============================================================

# Run with:
# uvicorn main:app --reload
