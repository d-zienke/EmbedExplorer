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
logging.basicConfig(level=logging.INFO, format='%(asctime=s - %(levelname=s - %(message=s')

# Load configuration from chatbot/config.json
with open('chatbot/config.json', 'r') as file:
    config = json.load(file)

# Initialize the model handler
model_handler = ModelHandler(config)

def generate_query_embedding(query_text):
    return model_handler.generate_embedding(query_text)

def retrieve_relevant_documents(query_text, config_path='vector_db/config.json'):
    with open(config_path, 'r') as file:
        vector_db_config = json.load(file)

    vector_db = VectorDatabase(vector_db_config)
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
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(read_document, file_paths))
    return results
