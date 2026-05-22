import os
import chromadb

# Define the directory path where ChromaDB will save data on your hard drive
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

# Ensure the data directory exists on your machine
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize the persistent ChromaDB client
# This ensures data is written to disk and survives server restarts
chroma_client = chromadb.PersistentClient(path=DATA_DIR)

def get_or_create_collection(collection_name="research_papers"):
    """
    Retrieves the existing collection or creates a new one if it doesn't exist.
    We override the default embedding function later during query/ingest operations.
    """
    return chroma_client.get_or_create_collection(name=collection_name)