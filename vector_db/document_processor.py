import os
import hashlib
from PyPDF2 import PdfReader
import markdown
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
import logging
from config import Config
import numpy as np
import faiss
import sqlite3

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')

class DocumentProcessor:
    """
    Class for processing documents and generating embeddings.
    """

    def __init__(self, vector_db, use_faiss=True):
        """
        Initialize the DocumentProcessor class.

        Args:
            vector_db (VectorDatabase): Instance of the VectorDatabase class.
            use_faiss (bool): Whether to use FAISS for storing embeddings. Useful for debugging.
        """
        self.vector_db = vector_db
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.use_faiss = use_faiss
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
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                text = "".join([page.extract_text() for page in reader.pages])
            logging.info(f"Extracted text from PDF: {file_path}")
            return text
        except Exception as e:
            logging.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""

    @staticmethod
    def extract_text_from_txt(file_path):
        """
        Extract text from a TXT file.

        Args:
            file_path (str): Path to the TXT file.

        Returns:
            str: Extracted text.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            logging.info(f"Extracted text from TXT: {file_path}")
            return text
        except Exception as e:
            logging.error(f"Error extracting text from TXT {file_path}: {e}")
            return ""

    @staticmethod
    def extract_text_from_markdown(file_path):
        """
        Extract text from a Markdown file.

        Args:
            file_path (str): Path to the Markdown file.

        Returns:
            str: Extracted text.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md = file.read()
            html = markdown.markdown(md)
            text = ''.join(re.findall(r'>[^<]+<', html))  # Extract text between tags
            logging.info(f"Extracted text from Markdown: {file_path}")
            return text
        except Exception as e:
            logging.error(f"Error extracting text from Markdown {file_path}: {e}")
            return ""

    @staticmethod
    def extract_title(text):
        """
        Extract the title from the text.

        Args:
            text (str): Text from which to extract the title.

        Returns:
            str: Extracted title.
        """
        try:
            title = text.splitlines()[0].strip() if text else "Untitled"
            logging.info(f"Extracted title: {title}")
            return title
        except Exception as e:
            logging.error(f"Error extracting title from text: {e}")
            return "Untitled"

    @staticmethod
    def chunk_text(text):
        """
        Chunk the text into smaller segments.

        Args:
            text (str): Text to be chunked.

        Returns:
            list: List of text chunks.
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            chunks = text_splitter.split_text(text)
            logging.info(f"Chunked text into {len(chunks)} segments.")
            return chunks
        except Exception as e:
            logging.error(f"Error chunking text: {e}")
            return []

    def generate_embeddings(self, text_chunks):
        """
        Generate embeddings for the text chunks.

        Args:
            text_chunks (list): List of text chunks.

        Returns:
            numpy.ndarray: List of embeddings.
        """
        try:
            embedding_vectors = self.model.encode(text_chunks, normalize_embeddings=True)
            logging.info(f"Generated {len(embedding_vectors)} embeddings with shape {embedding_vectors.shape}.")
            return embedding_vectors
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            return []

    def setup_sqlite(self):
        """ Setup SQLite connection and table. """
        conn = sqlite3.connect(Config.SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT,
                embedding BLOB
            )
        ''')
        conn.commit()
        return conn, cursor

    def store_embedding_in_sqlite_thread_safe(self, document_id, title, embedding):
        """ Store embedding in SQLite database from multiple threads. """
        try:
            conn, cursor = self.setup_sqlite()  # Create a new connection and cursor in each thread
            embedding_blob = embedding.tobytes()
            cursor.execute('''
                INSERT INTO documents (id, title, embedding) VALUES (?, ?, ?)
            ''', (document_id + "_" + str(np.random.randint(1000)), title, embedding_blob))
            conn.commit()
            cursor.close()
            conn.close()
            logging.info(f"Stored embedding in SQLite for document {document_id}.")
        except Exception as e:
            logging.error(f"Error storing embedding in SQLite: {e}")

    def process_document(self, file_path):
        """
        Process a document and store its embeddings in the database.

        Args:
            file_path (str): Path to the document.
        """
        try:
            ext = os.path.splitext(file_path)[-1].lower()
            if ext == ".pdf":
                text = self.extract_text_from_pdf(file_path)
            elif ext == ".txt":
                text = self.extract_text_from_txt(file_path)
            elif ext == ".md":
                text = self.extract_text_from_markdown(file_path)
            else:
                raise ValueError("Unsupported file type")

            if not text:
                logging.warning(f"No text extracted from {file_path}. Skipping this document.")
                return

            document_id = hashlib.md5(text.encode('utf-8')).hexdigest()
            title = self.extract_title(text)
            if not self.vector_db.is_document_processed(document_id):
                text_chunks = self.chunk_text(text)
                if text_chunks:
                    embeddings = self.generate_embeddings(text_chunks)
                    if isinstance(embeddings, np.ndarray) and len(embeddings) > 0:
                        if np.any(np.isnan(embeddings)):
                            logging.error(f"NaN values found in embeddings for document {file_path}.")
                        elif np.all(embeddings == 0):
                            logging.error(f"All-zero embeddings found for document {file_path}.")
                        elif embeddings.ndim == 2:  # Ensure embeddings are 2D
                            logging.debug(f"Embeddings are ready to be stored in FAISS with shape: {embeddings.shape}")
                            if self.use_faiss:
                                logging.info("Storing embeddings in FAISS index one by one.")
                                # Store embeddings one by one to identify issues
                                for i, embedding in enumerate(embeddings):
                                    try:
                                        self.store_embedding_in_sqlite_thread_safe(document_id + f'_{i}', title, embedding)
                                        self.vector_db.store_embeddings(document_id + f'_{i}', title, [embedding])
                                        logging.debug(f"Stored embedding {i + 1}/{len(embeddings)} successfully.")
                                    except Exception as e:
                                        logging.error(f"Error storing embedding {i + 1}/{len(embeddings)}: {e}")
                                logging.info("Marking document as processed in the database.")
                                self.vector_db.mark_document_as_processed(document_id, title, file_path)
                                logging.info(f"Processed document {file_path} with title '{title}' and stored embeddings.")
                            else:
                                logging.info("FAISS storage skipped (bypass mode).")
                        else:
                            logging.error(f"Embeddings for document {file_path} are not in the correct shape. Expected 2D, got {embeddings.ndim}D.")
                    else:
                        logging.error(f"Failed to generate valid embeddings for document {file_path}.")
                else:
                    logging.error(f"Failed to chunk text for document {file_path}.")
            else:
                logging.info(f"Document {file_path} has already been processed.")
        except Exception as e:
            logging.error(f"Error processing document {file_path}: {e}")
