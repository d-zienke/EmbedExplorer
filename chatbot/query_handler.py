# chatbot/query_handler.py

import json
from sentence_transformers import SentenceTransformer
from vector_db.database import VectorDatabase
import logging
import os
from PyPDF2 import PdfReader
import markdown
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname=s - %(message=s')

# Load configuration from chatbot/config.json
with open('chatbot/config.json', 'r') as file:
    config = json.load(file)

# Initialize SBERT model for embeddings
sbert_model = SentenceTransformer(config['sbert_model_name'])

def generate_query_embedding(query_text):
    """
    Generate an embedding for the query text using SBERT Multilingual.

    Args:
        query_text (str): The query text.

    Returns:
        np.ndarray: The embedding vector for the query text.
    """
    embedding = sbert_model.encode(query_text)
    return embedding

def retrieve_relevant_documents(query_text, config_path='vector_db/config.json'):
    """
    Retrieve the most relevant documents for the given query text.

    Args:
        query_text (str): The query text.
        config_path (str): Path to the configuration file.

    Returns:
        list: List of tuples containing document IDs and file paths.
    """
    # Load vector_db configuration
    with open(config_path, 'r') as file:
        vector_db_config = json.load(file)

    # Initialize the vector database
    vector_db = VectorDatabase(vector_db_config)

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

    # Close the vector database
    vector_db.close()

    return metadata

def read_document(file_path):
    """
    Read the content of a document.

    Args:
        file_path (str): Path to the document file.

    Returns:
        str: Content of the document.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
    elif ext == ".txt":
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    elif ext == ".md":
        with open(file_path, 'r', encoding='utf-8') as file:
            md = file.read()
        html = markdown.markdown(md)
        text = ''.join(re.findall(r'>[^<]+<', html))  # Extract text between tags
    else:
        raise ValueError("Unsupported file type")
    return text
