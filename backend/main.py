import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import our custom modules
from database import get_or_create_collection
from ingest import extract_text_from_pdf, chunk_text
from generate import get_embedding, generate_related_work

app = FastAPI(
    title="RAG Research Citation Assistant",
    description="Backend API for managing research papers and generating cited summaries."
)

# Enable CORS so your frontend UI can communicate with this backend smoothly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store uploaded files temporarily
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../uploaded_papers"))
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- PYDANTIC VALIDATION SCHEMAS ---

class QueryRequest(BaseModel):
    # Validation ensures the query string cannot be empty or pure whitespace
    query: str = Field(..., min_length=1, description="The research topic or abstract to query.")

class DeleteRequest(BaseModel):
    paper_title: str = Field(..., min_length=1, description="The exact title of the paper to delete.")


# --- API ROUTES ---

@app.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_paper(file: UploadFile = File(...)):
    """
    Route 1: Upload a PDF research paper.
    Parses, chunks, embeds, and stores the paper inside the persistent vector database.
    """
    # Strict validation: Check file extension
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Invalid file format. Only PDF files are supported."
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # Save uploaded binary stream to local disk temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Extract text from PDF
        raw_text = extract_text_from_pdf(file_path)
        if not raw_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="The uploaded PDF contains no extractable text."
            )

        # 2. Slice text into overlapping chunks
        chunks = chunk_text(raw_text)
        
        # 3. Process and upsert chunks to ChromaDB
        collection = get_or_create_collection()
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for i, chunk in enumerate(chunks):
            # Generate a unique string ID for each paragraph chunk
            chunk_id = f"{file.filename}_chunk_{i}"
            # Compute numerical vector meaning
            vector = get_embedding(chunk)
            
            ids.append(chunk_id)
            embeddings.append(vector)
            # Metadata must contain paper title to enforce zero-hallucination tracking
            metadatas.append({"paper_title": file.filename, "chunk_index": i})
            documents.append(chunk)

        # Bulk insert vectors and payloads into persistent storage
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

        return {"message": f"Successfully processed and indexed '{file.filename}' into vector database."}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error during ingestion: {str(e)}"
        )
    finally:
        # Clean up the temporary file from block storage
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/query")
async def query_library(request: QueryRequest):
    """
    Route 2: Execute semantic search and generate a cited research paper summary.
    """
    try:
        # Generate embedding vector for incoming text query
        query_vector = get_embedding(request.query)
        
        collection = get_or_create_collection()
        
        # Query ChromaDB for top 5 most contextually relevant matches (k-NN search)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=5
        )

        # Re-verify matching entries exist in our context store
        retrieved_chunks = []
        if results and results.get("documents") and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                retrieved_chunks.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                })

        if not retrieved_chunks:
            return {
                "query": request.query,
                "result": "No relevant reference documentation has been uploaded to the system library yet."
            }

        # 4. Generate the heavily grounded Related Work paragraph
        ai_response = generate_related_work(request.query, retrieved_chunks)
        
        return {
            "query": request.query,
            "result": ai_response
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error during query inference: {str(e)}"
        )


@app.get("/papers")
async def list_papers():
    """
    Route 3: List all unique research papers currently stored inside the vector database index.
    """
    try:
        collection = get_or_create_collection()
        existing_data = collection.get(include=["metadatas"])
        
        # Extract unique paper names out of chunk metadatas
        unique_titles = set()
        if existing_data and existing_data.get("metadatas"):
            for meta in existing_data["metadatas"]:
                if meta and "paper_title" in meta:
                    unique_titles.add(meta["paper_title"])

        return {"uploaded_papers": list(unique_titles)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch paper library index: {str(e)}"
        )


@app.delete("/paper")
async def delete_paper(request: DeleteRequest):
    """
    Route 4: Delete a specific research paper and all its associated structural vectors from disk.
    """
    try:
        collection = get_or_create_collection()
        
        # Target vectors matching metadata condition to avoid wiping the entire index
        collection.delete(where={"paper_title": request.paper_title})
        return {"message": f"Successfully deleted '{request.paper_title}' and flushed references from disk storage."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete specified data index: {str(e)}"
        )