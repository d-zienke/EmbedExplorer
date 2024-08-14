import os
import hashlib
from PyPDF2 import PdfReader
import markdown
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import logging
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class DocumentProcessor:
    """
    Class for processing documents and generating embeddings.
    """

    def __init__(self, vector_db):
        """
        Initialize the DocumentProcessor class.

        Args:
            vector_db (VectorDatabase): Instance of the VectorDatabase class.
        """
        self.vector_db = vector_db
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        logging.info(f"Loaded embedding model: {Config.EMBEDDING_MODEL}")

    @staticmethod
    def extract_text_from_pdf(file_path):
        """
        Extract text from a PDF file.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            str: Extracted text.
        """
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = "".join([page.extract_text() for page in reader.pages])
        logging.info(f"Extracted text from PDF: {file_path}")
        return text

    @staticmethod
    def extract_text_from_txt(file_path):
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

    @staticmethod
    def extract_text_from_markdown(file_path):
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

    @staticmethod
    def extract_title(text):
        """
        Extract the title from the text.

        Args:
            text (str): Text from which to extract the title.

        Returns:
            str: Extracted title.
        """
        # Simple title extraction: use the first line as the title
        title = text.splitlines()[0].strip()
        logging.info(f"Extracted title: {title}")
        return title

    @staticmethod
    def chunk_text(text):
        """
        Chunk the text into smaller segments.

        Args:
            text (str): Text to be chunked.

        Returns:
            list: List of text chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
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
        embedding_vectors = self.model.encode(text_chunks, convert_to_tensor=False)
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
        title = self.extract_title(text)
        if not self.vector_db.is_document_processed(document_id):
            text_chunks = self.chunk_text(text)
            embeddings = self.generate_embeddings(text_chunks)
            self.vector_db.store_embeddings(document_id, title, embeddings)
            self.vector_db.mark_document_as_processed(document_id, title, file_path)
            logging.info(f"Processed document {file_path} with title '{title}' and stored embeddings.")
