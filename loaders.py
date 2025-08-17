import os
from typing import List, Tuple
from pypdf import PdfReader

def read_text(path: str) ->str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()
    
def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_doc(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return read_text(path)
    elif ext == ".pdf":
        return read_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    
def chunk_text(text: str, chunk_size=800, overlap=150) -> List[Tuple[int, int, str]]:
    chunks = []
    n = len(text)
    i = 0
    while i < n:
        end = min(i + chunk_size, n)
        chunk = text[i:end]
        chunks.append((i, end, chunk))
        if end == n:
            break
        i = end - overlap
        if i < 0:
            i = 0
    return chunks