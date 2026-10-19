import os

import chromadb
import requests
from config import settings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="My RAG API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
)

# --- ChromaDB Setup ---
client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
collection = client.get_or_create_collection(
    settings.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
)


# --- Schemas ---
class AskRequest(BaseModel):
    question: str
    n_result: int = settings.MAX_RESULTS
    max_distance: float = settings.MAX_DISTANCE


class SourceChunk(BaseModel):
    text: str
    source: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    confidence: str


# --- RAG Logic ---
SYSTEM_PROMPT = """You are a helpful AI assistant who is a tutor for students learning AI and other software engineering skills.
Answer the user's query or prompts based ONLY on the context provided below.

Rules:
- Use only information provided in CONTEXT section below.
- If the context does not contain the answer, respond with 'I'm sorry, I don't have enough information to answer that. Please try a different query that falls within the content of the available documents.
- Cite your sources clearly by mentioning the document name in parenthases. 
- Keep your responses concise, under 200 words.
- Use clear analogies when applicaple to help your student understand the content better.
- NEVER make up information. If you do not have contextual information, respond with "I don't know the answer to that question, or use one of the hardcoded guardrail responses below."""


def retrieve(question: str, n_results: int, max_distance: float):
    if collection.count() == 0:
        return []
    results = collection.query(
        query_texts=[question], n_results=min(n_results, collection.count())
    )
    chunks = []
    for i in range(len(results["documents"][0])):
        dist = results["distances"][0][i]
        if dist <= max_distance:
            chunks.append(
                SourceChunk(
                    text=results["documents"][0][i],
                    source=results["metadatas"][0][i].get("source", "unknown"),
                    distance=round(dist, 4),
                )
            )
    return chunks


def get_confidence(chunks):
    if not chunks:
        return "none"
    best = chunks[0].distance
    if best < settings.CONFIDENCE_THRESHOLD:
        return "high"
    if best < settings.MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def generate(messages):
    try:
        r = requests.post(
            f"{settings.OLLAMA_URL}/api/chat",
            json={
                "model": settings.MODEL_NAME,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0},
            },
        )
        return r.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise HTTPException(503, "Ollama is not running")


# --- Endpoints ---
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="Please enter a question.")
    chunks = retrieve(req.question, req.n_result, req.max_distance)
    confidence = get_confidence(chunks)

    if not chunks:
        return AskResponse(
            answer="I don't have relevant information to answer that.",
            sources=[],
            confidence="none",
        )

    if confidence == "low":
        return AskResponse(
            answer=(
                "I found some loosely related information, but I'm not confident "
                "it answers your question. Try rephrasing or asking about a topic "
                "covered in the documents."
            ),
            sources=chunks,
            confidence=confidence,
        )

    context = "\n\n".join(f"[Source: {c.source}]\n{c.text}" for c in chunks)
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"},
        {"role": "user", "content": req.question},
    ]

    answer = generate(messages)
    return AskResponse(answer=answer, sources=chunks, confidence=confidence)


@app.post("/ingest")
def ingest_documents():
    docs_dir = settings.DOCS_DIRECTORY
    if not os.path.exists(docs_dir):
        raise HTTPException(404, f"{docs_dir}/ directory not found")

    chunks, ids, metadatas = [], [], []
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith((".txt", ".md")):
            continue
        with open(os.path.join(docs_dir, filename), "r") as f:
            content = f.read()
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs):
            chunks.append(para)
            ids.append(f"{filename}_{i}")
            metadatas.append({"source": filename, "chunk_index": str(i)})

    if chunks:
        collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)

    return {"message": f"Ingested {len(chunks)} chunks from {docs_dir}/"}


@app.get("/")
def root():
    return {"message": "My RAG API is running"}


@app.get("/stats")
def stats():
    return {
        "document_count": collection.count(),
        "olama_model": settings.MODEL_NAME,
        "db_path": settings.CHROMA_PATH,
    }


@app.get("/health")
def health():
    ollama_ok = False
    try:
        r = requests.get(f"{settings.OLLAMA_URL}/api/tags", timeout=2)
        ollama_ok = r.status_code == 200
    except requests.RequestException:
        ollama_ok = False
    return {
        "chromadb": "ok",
        "ollama": "ok" if ollama_ok else "unavailable",
        "documents": collection.count(),
    }


# Added comment to confirm caching
