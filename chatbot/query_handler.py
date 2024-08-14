import logging
import os
import re
import markdown
from PyPDF2 import PdfReader
from vector_db.database import VectorDatabase
from chatbot.model_handler import ModelHandler
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class QueryHandler:
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.model_handler = ModelHandler()

    def generate_query_embedding(self, query_text):
        """
        Generate an embedding for the query text using the embedding model.
        Args:
            query_text (str): The query text.
        Returns:
            np.ndarray: The embedding vector for the query text.
        """
        return self.model_handler.generate_embedding(query_text)

    def retrieve_relevant_documents(self, query_text, top_k=3):
        """
        Retrieve the most relevant documents for the given query text.
        Args:
            query_text (str): Query text to search for.
            top_k (int): Number of top relevant documents to retrieve.
        Returns:
            list: Metadata of the top relevant documents.
        """
        query_vector = self.generate_query_embedding(query_text)
        logging.info("Query vector generated.")
        top_indices = self.vector_db.query_embeddings(query_vector, top_k=top_k)
        logging.info(f"Top indices from FAISS query: {top_indices}")
        document_ids = [self.vector_db.index_to_id(idx) for idx in top_indices]
        logging.info(f"Document IDs retrieved: {document_ids}")
        metadata = self.vector_db.get_document_metadata(document_ids)
        logging.info(f"Metadata retrieved: {metadata}")
        return metadata

    def list_document_titles(self):
        """
        List all document titles stored in the vector database.
        Returns:
            list: List of document titles.
        """
        documents = self.vector_db.list_documents()
        titles = [title for _, title, _ in documents]
        logging.info(f"Retrieved document titles: {titles}")
        return titles


    @staticmethod
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
            return QueryHandler.extract_text_from_pdf(file_path)
        elif ext == ".txt":
            return QueryHandler.extract_text_from_txt(file_path)
        elif ext == ".md":
            return QueryHandler.extract_text_from_markdown(file_path)
        else:
            raise ValueError("Unsupported file type")

    @staticmethod
    def extract_text_from_pdf(file_path):
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = "".join([page.extract_text() for page in reader.pages])
        return text

    @staticmethod
    def extract_text_from_txt(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    @staticmethod
    def extract_text_from_markdown(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            md = file.read()
        html = markdown.markdown(md)
        return ''.join(re.findall(r'>[^<]+<', html))  # Extract text between tags

    def read_documents_in_parallel(self, file_paths):
        """
        Read multiple documents in parallel.
        Args:
            file_paths (list): List of file paths to read.
        Returns:
            list: List of text contents from the documents.
        """
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.read_document, file_paths))
        return results
