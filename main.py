from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import chromadb
import fitz  # PyMuPDF
from typing import List, Dict
from groq import Groq
import shutil

app = FastAPI()
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Chroma client
chroma_client = chromadb.Client()
COLLECTION_NAME = "my_collection"

# Initialize Groq client
client = Groq(
    api_key="YOUR_GROQ_API_KEY",
)

# Chunk text into smaller pieces
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Extract text from a PDF file
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

# Store chunks in ChromaDB
def store_in_chromadb(collection: chromadb.Collection, chunks: List[str], ids: List[str]) -> None:
    if len(chunks) != len(ids):
        raise ValueError("The number of chunks must match the number of IDs")
    collection.upsert(documents=chunks, ids=ids)

# Perform a vector search
def vector_search(collection: chromadb.Collection, query: str, top_k: int = 1) -> Dict[str, any]:
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG API"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        text = extract_text_from_pdf(file_location)
        chunks = chunk_text(text)
        ids = [f"id_{i}" for i in range(len(chunks))]
        
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"} )
        store_in_chromadb(collection, chunks, ids)
        
        return {"filename": file.filename, "chunks": len(chunks)}
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"Error processing file: {str(e)}"})

@app.get("/files/")
def list_files():
    files = os.listdir(UPLOAD_FOLDER)
    return {"files": files}

@app.delete("/files/{filename}")
async def delete_file(filename: str):
    file_location = os.path.join(UPLOAD_FOLDER, filename)
    file_location = os.path.normpath(file_location)

    if os.path.isfile(file_location):
        try:
            os.remove(file_location)
            return JSONResponse(status_code=200, content={"message": "File deleted"})
        except Exception as e:
            return JSONResponse(status_code=500, content={"message": f"Failed to delete file: {str(e)}"})
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/search/")
def search(query: str):
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    results = vector_search(collection, query)
    return results

@app.get("/llmquery/")
def llm_query(query: str):
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    context = vector_search(collection, query)
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Provide responses based solely on the following context. Do not answer questions outside of this context. If the user asks something irrelevant or outside the context, respond with 'I don't know.'",
            },
            {
                "role": "system",
                "content": str(context),
            },
            {
                "role": "user",
                "content": query,
            }
        ],
        model="llama3-8b-8192",
    )
    return chat_completion

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
