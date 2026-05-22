import fitz  # PyMuPDF library
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(file_path: str) -> str:
    """
    Opens a PDF file using PyMuPDF and extracts all text page by page.
    """
    text = ""
    try:
        # Open the PDF document
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error parsing PDF at {file_path}: {e}")
        raise e
    return text

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Splits long strings of text into manageable, overlapping paragraphs.
    Using standard 1000 character windows with a 200 character overlap 
    ensures structural context isn't chopped in half at chunk boundaries.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    # Split text into a list of plain strings
    chunks = text_splitter.split_text(text)
    return chunks