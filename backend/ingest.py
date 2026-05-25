import os
import re
import logging
import fitz  # PyMuPDF library
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure basic logging architecture to avoid raw stdout streaming
logging.basicConfig(level=logging.INFO)

def extract_text_with_metadata(file_path: str):
    """
    Opens a PDF file using PyMuPDF, extracts text, captures cover page 
    academic metadata, and tracks section boundaries line-by-line.
    """
    raw_text = ""
    documents_by_section = []
    filename = os.path.basename(file_path)
    
    try:
        # Open the PDF document stream using PyMuPDF
        doc = fitz.open(file_path)
        for page in doc:
            raw_text += page.get_text()
        doc.close()
    except Exception as e:
        # --- FIX 4: REPLACE DEBUG PRINT WITH LOGGING.ERROR ---
        logging.error(f"Error parsing PDF at {file_path}: {e}")
        raise e

    # --- V2 FEATURE: ACADEMIC METADATA EXTRACTION (FIRST 2,000 CHARACTERS) ---
    cover_slice = raw_text[:2000]
    
    # 1. Title Extraction (Fallback to filename if unstructured)
    extracted_title = filename
    
    # 2. Publishing Year Extraction via 4-digit boundary regex filter
    year_match = re.search(r'\b(19|20)\d{2}\b', cover_slice)
    extracted_year = year_match.group(0) if year_match else "Unknown Year"
    
    # 3. Author Extraction (Fallback safety rule to prevent runtime crashes)
    # Searches for common structural indicators or lines below title space
    author_match = re.search(r'(?i)by:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', cover_slice)
    extracted_author = author_match.group(1) if author_match else "Unknown Author"

    # --- V2 FEATURE: LINE-BY-LINE STRUCTURAL SECTION SEGMENTATION ---
    sections_re = re.compile(r'^(abstract|introduction|methodology|results|discussion|conclusion)s?$', re.IGNORECASE)
    
    current_section = "general"
    section_buffer = []

    # Walk through line-by-line to capture layout tracking transitions
    for line in raw_text.split('\n'):
        clean_line = line.strip()
        if not clean_line:
            continue
            
        # Check if the line matches an academic heading boundary
        if sections_re.match(clean_line):
            # Save the accumulated text from the previous section before switching states
            if section_buffer:
                documents_by_section.append({
                    "text": " ".join(section_buffer),
                    "section": current_section
                })
                section_buffer = []
            current_section = clean_line.lower()
        else:
            section_buffer.append(clean_line)

    # Append any remaining residual text left in the final buffer chunk
    if section_buffer:
        documents_by_section.append({
            "text": " ".join(section_buffer),
            "section": current_section
        })

    # Pack global file-level tracking indices together
    global_metadata = {
        "filename": filename,
        "title": extracted_title,
        "author": extracted_author,
        "year": extracted_year
    }

    return documents_by_section, global_metadata

def chunk_section_text(documents_by_section, global_metadata, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Splits text blocks logically grouped by sections into manageable chunks, 
    appending the full composite metadata dictionary payloads for ChromaDB.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    final_processed_chunks = []

    # Processes text chunks sequentially without crossing structural section lines
    for doc in documents_by_section:
        chunks = text_splitter.split_text(doc["text"])
        
        for chunk in chunks:
            # Construct the comprehensive multi-key payload dictionary
            chunk_payload = {
                "text": chunk,
                "metadata": {
                    "paper_title": global_metadata["filename"],
                    "extracted_title": global_metadata["title"],
                    "author": global_metadata["author"],
                    "year": global_metadata["year"],
                    "section": doc["section"]  # Links the chunk directly to its visual UI badge label
                }
            }
            final_processed_chunks.append(chunk_payload)

    return final_processed_chunks


# --- 🛠️ THE FIX: ADDING THE EXACT ROUTE LINKS YOUR MAIN.PY IS LOOKING FOR ---

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts and returns all raw text strings from a PDF file for main.py."""
    raw_text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            raw_text += page.get_text()
        doc.close()
        return raw_text
    except Exception as e:
        logging.error(f"Error parsing PDF at {file_path}: {e}")
        raise e

def chunk_text(text: str) -> list:
    """Splits section string data into text fragments for main.py."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)