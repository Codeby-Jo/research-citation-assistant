import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# 1. Keep your free local text embeddings intact
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Re-point the client to the global OpenRouter gateway
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://127.0.0.1:5500", # Helps OpenRouter track app performance
        "X-OpenRouter-Title": "RAG Citation Assistant"
    }
)

def get_embedding(text: str) -> list:
    """Generates a free vector embedding locally using sentence-transformers."""
    try:
        return embedding_model.encode(text).tolist()
    except Exception as e:
        print(f"Error generating local embedding: {e}")
        raise e

def generate_related_work(query: str, retrieved_chunks: list) -> str:
    """Compiles chunks and requests synthesis from a free OpenRouter model."""
    context_block = ""
    for idx, chunk in enumerate(retrieved_chunks):
        title = chunk.get("metadata", {}).get("paper_title", "Unknown Document")
        text = chunk.get("text", "")
        context_block += f"[Source {idx+1}: {title}]\n{text}\n\n"

    system_prompt = (
        "You are an elite academic research citation assistant.\n"
        "Your task is to draft a 2-3 paragraph 'Related Work' or 'Literature Review' summary "
        "addressing the user's research topic based exclusively on the provided source texts.\n\n"
        "UNBREAKABLE CONSTRAINTS:\n"
        "1. Synthesize the findings into a cohesive narrative. Do not just list the papers.\n"
        "2. You must cite source papers directly by their exact titles inline when referencing their work.\n"
        "3. Absolutely NO external knowledge or fabrication. Only cite papers that appear in the text below.\n"
        "4. If the provided documents are completely unrelated to the topic, state clearly that no relevant "
        "literature was found in the provided library."
    )

    user_prompt = f"User Research Topic: {query}\n\nRetrieved Source Texts:\n{context_block}"

    try:
        # Utilizing OpenRouter's free automated model router tag
        response = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free"),, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during OpenRouter text generation: {e}")
        raise e