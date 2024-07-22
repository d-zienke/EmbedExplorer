import json
import numpy as np
from sentence_transformers import SentenceTransformer
from vector_db.database import VectorDatabase
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname=s - %(message)s')

# Initialize SBERT model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def generate_query_embedding(query_text):
    """
    Generate an embedding for the query text using SBERT Multilingual.

    Args:
        query_text (str): The query text.

    Returns:
        np.ndarray: The embedding vector for the query text.
    """
    embedding = model.encode(query_text)
    return embedding

def main():
    # Load configuration
    with open('vector_db/config.json', 'r') as file:
        config = json.load(file)

    # Initialize the vector database
    vector_db = VectorDatabase(config)

    # Example query text
    query_text = "little golden key in the lock"

    # Generate the query embedding
    query_vector = generate_query_embedding(query_text)
    logging.info(f"Query vector generated.")

    # Perform the query
    top_indices = vector_db.query_embeddings(query_vector, top_k=3)
    logging.info(f"Top indices from FAISS query: {top_indices}")

    # Retrieve metadata for the top results
    document_ids = [vector_db.index_to_id(idx) for idx in top_indices]
    logging.info(f"Document IDs retrieved: {document_ids}")

    metadata = vector_db.get_document_metadata(document_ids)
    logging.info(f"Metadata retrieved: {metadata}")

    # Print the results
    for doc_id, file_path in metadata:
        print(f"Document ID: {doc_id}, File Path: {file_path}")

    # Close the vector database
    vector_db.close()

if __name__ == "__main__":
    main()
