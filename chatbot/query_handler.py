import json
import logging
from vector_db.database import VectorDatabase
from chatbot.model_handler import ModelHandler
import os
from PyPDF2 import PdfReader
import markdown
import re
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the model handler
model_handler = ModelHandler()


def generate_query_embedding(query_text):
    """
    Generate an embedding for the query text using SBERT Multilingual.

    Args:
        query_text (str): The query text.

    Returns:
        np.ndarray: The embedding vector for the query text.
    """
    return model_handler.generate_embedding(query_text)


def retrieve_relevant_documents(query_text):
    """
    Retrieve the most relevant documents for the given query text.

    Args:
        query_text (str): Query text to search for.

    Returns:
        list: Metadata of the top relevant documents.
    """
    vector_db = VectorDatabase()
    query_vector = generate_query_embedding(query_text)
    logging.info(f"Query vector generated.")
    top_indices = vector_db.query_embeddings(query_vector, top_k=3)
    logging.info(f"Top indices from FAISS query: {top_indices}")
    document_ids = [vector_db.index_to_id(idx) for idx in top_indices]
    logging.info(f"Document IDs retrieved: {document_ids}")
    metadata = vector_db.get_document_metadata(document_ids)
    logging.info(f"Metadata retrieved: {metadata}")
    vector_db.close()
    return metadata


def read_document(file_path):
    """
    Read the contents of a document from the specified file path.

    Args:
        file_path (str): Path to the document file.

    Returns:
        str: Extracted text content from the document.
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


def read_documents_in_parallel(file_paths):
    """
    Read multiple documents in parallel.

    Args:
        file_paths (list): List of file paths to read.

    Returns:
        list: List of text contents from the documents.
    """
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(read_document, file_paths))
    return results
