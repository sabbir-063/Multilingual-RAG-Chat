import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pipeline import RAGPipeline
from dotenv import load_dotenv
import shutil

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY missing in .env")

# Initialize RAG pipeline
rag = RAGPipeline(api_key=API_KEY)
# Load existing index if present
if os.path.exists(rag.store.index_path) and os.path.exists(rag.store.meta_path):
    rag.load_index()

app = FastAPI(title="RAG FastAPI Demo")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------
# Upload and index files
# ----------------------------
@app.post("/index/")
async def index_files(files: list[UploadFile] = File(...)):
    file_paths = []
    for f in files:
        save_path = os.path.join(UPLOAD_DIR, f.filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        file_paths.append(save_path)

    rag.index_files(file_paths)
    return {"message": f"Indexed {len(file_paths)} files successfully."}

# ----------------------------
# Ask a question
# ----------------------------
@app.get("/ask/")
async def ask_question(query: str, k: int = 3):
    answer, sources = rag.ask(query, k=k)
    source_info = [
        {"source": c.source, "text": len(c.text), "score": s} for c, s in sources
    ]
    return {"answer": answer, "sources": source_info}
