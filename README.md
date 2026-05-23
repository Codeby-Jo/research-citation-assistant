# RAG Research Citation Assistant

A Retrieval-Augmented Generation application that allows academic researchers to upload a library of PDFs and instantly generate cited literature summaries.

## Tech Stack
- **Backend:** FastAPI, Pydantic
- **Vector Store:** ChromaDB (Persistent Local Disk)
- **Embeddings:** Local Sentence-Transformers (`all-MiniLM-L6-v2`)
- **LLM Synthesis:** OpenRouter API (`openrouter/free` auto-router)
- **Frontend:** Vanilla HTML5, Tailwind CSS, JavaScript Fetch API

## Fresh Machine Installation Steps
1. Clone the repository and navigate to the project directory.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate

### ⚙️ Environment Variables Setup (V2 Upgrades)

Create a `.env` file in the root directory of your project space and specify the following parameters:

```text
OPENAI_API_KEY="your_secret_openrouter_api_key_here"
LLM_MODEL="openrouter/free"
RELEVANCE_THRESHOLD=1.5