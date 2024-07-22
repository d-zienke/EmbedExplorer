import os
import hashlib
from PyPDF2 import PdfReader
import markdown
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ollama import embeddings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentProcessor:
    """
    Class for processing documents and generating embeddings.
    """

    def __init__(self, config, vector_db):
        """
        Initialize the DocumentProcessor class.

        Args:
            config (dict): Configuration dictionary.
            vector_db (VectorDatabase): Instance of the VectorDatabase class.
        """
        self.config = config
        self.vector_db = vector_db

    def extract_text_from_pdf(self, file_path):
        """
        Extract text from a PDF file.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            str: Extracted text.
        """
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            logging.info(f"Extracted text from PDF: {file_path}")
            return text

    def extract_text_from_txt(self, file_path):
        """
        Extract text from a TXT file.

        Args:
            file_path (str): Path to the TXT file.

        Returns:
            str: Extracted text.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        logging.info(f"Extracted text from TXT: {file_path}")
        return text

    def extract_text_from_markdown(self, file_path):
        """
        Extract text from a Markdown file.

        Args:
            file_path (str): Path to the Markdown file.

        Returns:
            str: Extracted text.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            md = file.read()
        html = markdown.markdown(md)
        text = ''.join(re.findall(r'>[^<]+<', html))  # Extract text between tags
        logging.info(f"Extracted text from Markdown: {file_path}")
        return text

    def chunk_text(self, text):
        """
        Chunk the text into smaller segments.

        Args:
            text (str): Text to be chunked.

        Returns:
            list: List of text chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"]
        )
        chunks = text_splitter.split_text(text)
        logging.info(f"Chunked text into {len(chunks)} segments.")
        return chunks

    def generate_embeddings(self, text_chunks):
        """
        Generate embeddings for the text chunks.

        Args:
            text_chunks (list): List of text chunks.

        Returns:
            list: List of embeddings.
        """
        embedding_vectors = []
        for chunk in text_chunks:
            result = embeddings(model=self.config["embedding_model"], prompt=f"Represent this sentence for searching relevant passages: {chunk}")
            vector = result["embedding"]
            embedding_vectors.append(vector)
        logging.info(f"Generated {len(embedding_vectors)} embeddings.")
        return embedding_vectors

    def process_document(self, file_path):
        """
        Process a document and store its embeddings in the database.

        Args:
            file_path (str): Path to the document.
        """
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".pdf":
            text = self.extract_text_from_pdf(file_path)
        elif ext == ".txt":
            text = self.extract_text_from_txt(file_path)
        elif ext == ".md":
            text = self.extract_text_from_markdown(file_path)
        else:
            raise ValueError("Unsupported file type")

        document_id = hashlib.md5(text.encode('utf-8')).hexdigest()
        if not self.vector_db.is_document_processed(document_id):
            text_chunks = self.chunk_text(text)
            embeddings = self.generate_embeddings(text_chunks)
            self.vector_db.store_embeddings(document_id, embeddings)
            self.vector_db.mark_document_as_processed(document_id, file_path)
            logging.info(f"Processed document {file_path} and stored embeddings.")
