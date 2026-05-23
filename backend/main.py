import os
import re
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import our custom modules
from database import get_or_create_collection
from ingest import extract_text_from_pdf, chunk_text
from generate import get_embedding, generate_related_work

app = FastAPI(
    title="RAG Research Citation Assistant - V2",
    description="Backend API for managing research papers with metadata processing, section-aware splitting, and distance tracking filters."
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


# --- INTERNAL PROCESSING HELPER FUNCTIONS (V2 UPGRADES) ---

def extract_academic_metadata(text: str, filename: str):
    """
    Upgrade 1 Fallback Parser: Scans the first page layout text to extract 
    Title, Authors, and Publication Year. Falls back to filename if undetected.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    # Fallback initialization rules
    title = filename
    authors = "Unknown Author"
    year = "Unknown Year"
    
    if lines:
        title = lines[0][:150]  # Grab primary header line as candidate title

    # Regex scan for standard 4-digit publication years (1990-2029)
    year_match = re.search(r"\b(19|20)\d{2}\b", text)
    if year_match:
        year = year_match.group(0)

    # Secondary line heuristic evaluation for author names
    if len(lines) > 1 and "abstract" not in lines[1].lower() and len(lines[1]) < 200:
        authors = lines[1]

    return title, authors, year


def perform_section_aware_chunking(raw_text: str):
    """
    Upgrade 4 Parser: Segments continuous document blocks by technical headings 
    before parsing individual character limit thresholds inside them.
    """
    # Look for standard case-insensitive academic structure boundaries
    heading_pattern = re.compile(
        r"^(Abstract|Introduction|Related Work|Literature Review|Methodology|Methods|Results|Discussion|Conclusion|References)", 
        re.IGNORECASE
    )
    
    lines = raw_text.split("\n")
    current_section = "Introduction"  # Core fallback block
    section_map = {current_section: []}
    
    for line in lines:
        line_clean = line.strip()
        match = heading_pattern.match(line_clean)
        
        # Enforce short-string validation to ensure lines are headings, not body text
        if match and len(line_clean) < 60:
            current_section = match.group(1).capitalize()
            if current_section not in section_map:
                section_map[current_section] = []
        else:
            section_map[current_section].append(line)
            
    section_aware_chunks = []
    for section_name, section_lines in section_map.items():
        section_body = "\n".join(section_lines).strip()
        if section_body:
            # Execute standard character split overlap metrics safely inside this section boundary
            sub_chunks = chunk_text(section_body)
            for text_segment in sub_chunks:
                section_aware_chunks.append({
                    "text": text_segment,
                    "section": section_name.lower()
                })
                
    return section_aware_chunks


# --- PYDANTIC VALIDATION SCHEMAS ---

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The research topic or abstract to query.")

class DeleteRequest(BaseModel):
    paper_title: str = Field(..., min_length=1, description="The exact title of the paper to delete.")


# --- API ROUTES ---

@app.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_paper(file: UploadFile = File(...)):
    """
    Route 1: Upload a PDF research paper with Duplicate Prevention (Upgrade 3),
    Citation Extraction (Upgrade 1), and Section Scanning (Upgrade 4).
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Invalid file format. Only PDF files are supported."
        )

    # --- UPGRADE 3: DUPLICATE DETECTION CHECK ---
    collection = get_or_create_collection()
    existing_check = collection.get(where={"paper_title": file.filename}, limit=1)
    if existing_check and existing_check.get("ids"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This paper has already been uploaded. Delete it first if you want to re-index it."
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        raw_text = extract_text_from_pdf(file_path)
        if not raw_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="The uploaded PDF contains no extractable text."
            )

        # --- UPGRADE 1: EXTRACT CITATION METADATA FIELDS ---
        extracted_title, extracted_authors, extracted_year = extract_academic_metadata(raw_text[:2000], file.filename)

        # --- UPGRADE 4: SEGMENT VIA SECTION BOUNDARIES ---
        section_chunks = perform_section_aware_chunking(raw_text)
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for i, chunk_data in enumerate(section_chunks):
            chunk_id = f"{file.filename}_chunk_{i}"
            vector = get_embedding(chunk_data["text"])
            
            ids.append(chunk_id)
            embeddings.append(vector)
            
            # Save the robust multi-layered metadata parameters to ChromaDB
            metadatas.append({
                "paper_title": file.filename, 
                "extracted_title": extracted_title,
                "author": extracted_authors,
                "year": extracted_year,
                "section": chunk_data["section"],
                "chunk_index": i
            })
            documents.append(chunk_data["text"])

        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

        return {
            "message": f"Successfully processed '{file.filename}'.",
            "metadata": {"title": extracted_title, "author": extracted_authors, "year": extracted_year}
        }

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error during ingestion: {str(e)}"
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/query")
async def query_library(request: QueryRequest):
    """
    Route 2: Execute semantic search with Relevance Distance Threshold Filtering (Upgrade 2).
    """
    try:
        query_vector = get_embedding(request.query)
        collection = get_or_create_collection()
        
        # --- UPGRADE 2: REQUEST DISTANCE SCORES FROM CHROMADB ---
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )

        # Retrieve dynamic environment threshold with production manual fallback
        distance_threshold = float(os.environ.get("RELEVANCE_THRESHOLD", 0.7))

        retrieved_chunks = []
        if results and results.get("documents") and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(documents)
            
            for i in range(len(documents)):
                # Filter indices that stay within our similarity proximity boundary
                if distances[i] <= distance_threshold:
                    retrieved_chunks.append({
                        "text": documents[i],
                        "metadata": metadatas[i],
                        "distance": float(distances[i])
                    })

        # Return a clear message if no content satisfies the similarity criteria
        if not retrieved_chunks:
            return {
                "query": request.query,
                "result": "No sufficiently relevant papers were found in your library for this query."
            }

        # Generate the synthesis using our validated relevance segments
        ai_response = generate_related_work(request.query, retrieved_chunks)
        
        return {
            "query": request.query,
            "result": ai_response,
            "context_chunks": retrieved_chunks
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error during query inference: {str(e)}"
        )


@app.get("/papers")
async def list_papers():
    """
    Route 3: List all research papers with their corresponding extracted metadata.
    """
    try:
        collection = get_or_create_collection()
        existing_data = collection.get(include=["metadatas"])
        
        papers_dict = {}
        if existing_data and existing_data.get("metadatas"):
            for meta in existing_data["metadatas"]:
                if meta and "paper_title" in meta:
                    title_key = meta["paper_title"]
                    if title_key not in papers_dict:
                        papers_dict[title_key] = {
                            "filename": title_key,
                            "title": meta.get("extracted_title", title_key),
                            "author": meta.get("author", "Unknown Author"),
                            "year": meta.get("year", "Unknown Year")
                        }

        return {"uploaded_papers": list(papers_dict.values())}
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
        collection.delete(where={"paper_title": request.paper_title})
        return {"message": f"Successfully deleted '{request.paper_title}' and flushed references from disk storage."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete specified data index: {str(e)}"
        )