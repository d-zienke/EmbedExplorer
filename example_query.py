import json
import numpy as np
from vector_db.database import VectorDatabase

def generate_query_embedding(query_text):
    """
    Generate an embedding for the query text.
    This function is a placeholder and should be replaced with the actual embedding generation code.

    Args:
        query_text (str): The query text.

    Returns:
        np.ndarray: The embedding vector for the query text.
    """
    # Placeholder: Replace with your actual embedding generation code
    # For demonstration purposes, we'll use a random vector
    return np.random.rand(1024).astype(np.float32)

def main():
    # Load configuration
    with open('vector_db/config.json', 'r') as file:
        config = json.load(file)

    # Initialize the vector database
    vector_db = VectorDatabase(config)

    # Example query text
    query_text = "What is LLM?"

    # Generate the query embedding
    query_vector = generate_query_embedding(query_text)
    print(f"Query vector generated: {query_vector}")

    # Perform the query
    top_indices = vector_db.query_embeddings(query_vector, top_k=3)
    print(f"Top indices from FAISS query: {top_indices}")

    # Retrieve metadata for the top results
    document_ids = [vector_db.index_to_id(idx) for idx in top_indices]
    print(f"Document IDs retrieved: {document_ids}")

    metadata = vector_db.get_document_metadata(document_ids)
    print(f"Metadata retrieved: {metadata}")

    # Print the results
    for doc_id, file_path in metadata:
        print(f"Document ID: {doc_id}, File Path: {file_path}")

    # Close the vector database
    vector_db.close()

if __name__ == "__main__":
    main()
