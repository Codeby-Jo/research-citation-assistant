import os
import logging
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# --- CODE HYGIENE UPGRADE: CONFIGURE LOGGER INTERFACE ---
# Establishes structural tracking inside the file stream instead of dropping unlogged stdout lines
logger = logging.getLogger(__name__)

# 1. Keep your free local text embeddings intact
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Re-point the client to the global OpenRouter gateway
# --- FIX 4: REMOVED HARDCODED VS CODE LIVE SERVER PORT DEV ARTIFACT ---
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "X-OpenRouter-Title": "RAG Citation Assistant"
    }
)

def get_embedding(text: str) -> list:
    """Generates a free vector embedding locally using sentence-transformers."""
    try:
        return embedding_model.encode(text).tolist()
    except Exception as e:
        # --- FIX 4: UPGRADED DEBUG PRINT TO STRUCTURED LOGGING ---
        logging.error(f"Error generating local embedding: {e}", exc_info=True)
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
        # Resolves the stable model ID dynamically from your environment configuration
        selected_model = os.environ.get("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
        
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
        logging.error(f"Error during OpenRouter text generation: {e}", exc_info=True)
        raise e