import os
import logging
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- CODE HYGIENE UPGRADE: CONFIGURE MODULE-LEVEL LOGGER INTERFACE ---
logger = logging.getLogger(__name__)

# 🔥 FIX 1: BULLETPROOF PATHING - Locates the .env file dynamically in the project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)

# 🔥 FIX 2: MODEL INITIALIZATION - Loads the local embedding architecture into system memory
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Retrieve the API key string from the environment
api_key = os.environ.get("OPENAI_API_KEY")

# Safety guard checking to make sure your keys loaded cleanly
if not api_key:
    logger.warning("⚠️ OPENAI_API_KEY is empty! Double-check that your '.env' file position is correct.")

# Initialize OpenAI client by explicitly assigning the loaded key string
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "X-OpenRouter-Title": "RAG Research Citation Assistant"
    }
)

def get_embedding(text: str) -> list:
    """Generates a free vector embedding locally using sentence-transformers."""
    try:
        return embedding_model.encode(text).tolist()
    except Exception as e:
        # --- FIX 4: UPGRADED DEBUG PRINT TO STRUCTURED LOGGING ---
        logger.error(f"Error generating local embedding layer: {str(e)}", exc_info=True)
        raise e

def generate_related_work(query: str, retrieved_chunks: list) -> str:
    """
    Upgrade 1 & 4 Generation Block: Compiles context text along with academic 
    metadata boundaries to synthesize structured inline citations.
    """
    context_block = ""
    for idx, chunk in enumerate(retrieved_chunks):
        metadata = chunk.get("metadata", {})
        
        # Extract rich metadata tracking payloads with safe structural fallbacks
        filename = metadata.get("paper_title", "Unknown Document")
        title = metadata.get("extracted_title", filename)
        author = metadata.get("author", "Unknown Author")
        year = metadata.get("year", "Unknown Year")
        section = metadata.get("section", "unknown section")
        
        text = chunk.get("text", "")
        
        # Construct a highly visible data payload for the LLM to process
        context_block += (
            f"[Source {idx+1}]\n"
            f"Filename: {filename}\n"
            f"Extracted Title: {title}\n"
            f"Author/s: {author}\n"
            f"Year: {year}\n"
            f"Document Section: {section}\n"
            f"Content: {text}\n\n"
        )

    system_prompt = (
        "You are an elite academic research citation assistant.\n"
        "Your task is to draft a clean, professional 2-3 paragraph 'Related Work' or 'Literature Review' summary "
        "addressing the user's research topic based exclusively on the provided source texts.\n\n"
        "UNBREAKABLE CITATION CONSTRAINTS:\n"
        "1. Synthesize the findings into a fluid academic narrative. Do not just list or summarize papers one by one.\n"
        "2. MANDATORY CITATION FORMAT: You must cite source papers inline using exactly this format: (Author et al., Year - Paper Title).\n"
        "   - Example: 'Prior work on attention mechanisms (Vaswani et al., 2017 - Attention Is All You Need) demonstrated...'\n"
        "3. CRITICAL: Never use filenames (e.g., 'BIG DATA.pdf') as citations. Use the provided Extracted Title, Author/s, and Year attributes.\n"
        "4. Absolutely NO external knowledge lookup or fabrication. Only reference facts and papers that appear explicitly in the text block below.\n"
        "5. If the provided documents are completely unrelated to the topic, state clearly that no relevant literature was found in the provided library."
    )

    user_prompt = f"User Research Topic: {query}\n\nRetrieved Source Texts:\n{context_block}"

    try:
        # 🔥 FIX 3: OPENROUTER 404 RESOLUTION - Uses the stable, global free routing slug
        selected_model = os.environ.get("LLM_MODEL", "openrouter/free")
        
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        # --- FIX 4: UPGRADED DEBUG PRINT TO STRUCTURED LOGGING ---
        logger.error(f"Error during OpenRouter text generation inference sequence: {str(e)}", exc_info=True)
        raise e