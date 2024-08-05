import os
import sqlite3
import faiss
import numpy as np
import pickle
import logging
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VectorDatabase:
    """
    Class for managing the SQLite and FAISS database operations.
    """

    def __init__(self, config=None):
        """
        Initialize the VectorDatabase class.

        Args:
            config (dict or Config): Configuration dictionary or Config object. If not provided, defaults to Config.
        """
        if config is None:
            config = Config
        elif isinstance(config, dict):
            # Allow dict-like access for dictionary config
            config = type('Config', (object,), config)

        self.dimension = config.EMBEDDING_DIMENSION  # Dimension of the embeddings used
        self.conn = None
        self.cursor = None
        self.id_to_index = {}
        self.index_to_id_map = {}
        self.sqlite_db_path = config.SQLITE_DB_PATH
        self.faiss_index_path = config.FAISS_INDEX_PATH
        self.index = None  # Initialize self.index here
        self.setup_sqlite()
        self.setup_faiss()
        self.load_mappings()

    def setup_sqlite(self):
        """
        Set up the SQLite database connection and cursor.
        """
        try:
            self.conn = sqlite3.connect(self.sqlite_db_path)
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    file_path TEXT,
                    status TEXT
                )
            ''')
            self.conn.commit()
            logging.info("SQLite setup complete.")
        except sqlite3.Error as e:
            logging.error(f"SQLite setup error: {e}")
            raise

    def setup_faiss(self):
        """
        Set up the FAISS index.
        """
        try:
            if os.path.exists(self.faiss_index_path):
                self.index = faiss.read_index(self.faiss_index_path)
                if self.index.d != self.dimension:
                    logging.warning("Existing FAISS index dimension does not match. Reinitializing index.")
                    self.index = faiss.IndexFlatL2(self.dimension)
                    faiss.write_index(self.index, self.faiss_index_path)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
            logging.info("FAISS setup complete.")
        except Exception as e:
            logging.error(f"FAISS setup error: {e}")
            raise

    def clear_database(self):
        """
        Clear the SQLite and FAISS databases.
        """
        try:
            self.cursor.execute("DELETE FROM documents")
            self.conn.commit()
            self.index.reset()
            faiss.write_index(self.index, self.faiss_index_path)
            self.id_to_index.clear()
            self.index_to_id_map.clear()
            self.save_mappings()
            logging.info("Database cleared.")
        except Exception as e:
            logging.error(f"Error clearing database: {e}")
            raise

    def store_embeddings(self, document_id, embeddings):
        """
        Store the embeddings in the FAISS index.

        Args:
            document_id (str): Document ID.
            embeddings (list): List of embeddings.
        """
        try:
            embeddings_array = np.array(embeddings).astype(np.float32)
            if embeddings_array.ndim == 1:
                embeddings_array = embeddings_array.reshape(1, -1)
            assert embeddings_array.shape[1] == self.dimension, \
                f"Embedding dimension mismatch: {embeddings_array.shape[1]} != {self.dimension}"
            start_index = self.index.ntotal
            self.index.add(embeddings_array)
            for i in range(embeddings_array.shape[0]):
                index = start_index + i
                self.id_to_index[document_id] = index
                self.index_to_id_map[index] = document_id
            faiss.write_index(self.index, self.faiss_index_path)
            self.save_mappings()
            logging.info(f"Embeddings for document {document_id} stored.")
        except Exception as e:
            logging.error(f"Error storing embeddings for document {document_id}: {e}")
            raise

    def is_document_processed(self, document_id):
        """
        Check if the document is already processed.

        Args:
            document_id (str): Document ID.

        Returns:
            bool: True if the document is processed, False otherwise.
        """
        try:
            self.cursor.execute("SELECT status FROM documents WHERE id=?", (document_id,))
            result = self.cursor.fetchone()
            return result is not None
        except sqlite3.Error as e:
            logging.error(f"Error checking document {document_id}: {e}")
            raise

    def mark_document_as_processed(self, document_id, file_path):
        """
        Mark the document as processed.

        Args:
            document_id (str): Document ID.
            file_path (str): File path of the document.
        """
        try:
            if not self.is_document_processed(document_id):
                self.cursor.execute("INSERT INTO documents (id, file_path, status) VALUES (?, ?, ?)",
                                    (document_id, file_path, 'processed'))
                self.conn.commit()
                logging.info(f"Document {document_id} marked as processed.")
        except sqlite3.Error as e:
            logging.error(f"Error marking document {document_id} as processed: {e}")
            raise

    def list_documents(self):
        """
        List all processed documents.

        Returns:
            list: List of tuples containing document IDs and file paths.
        """
        try:
            self.cursor.execute("SELECT id, file_path FROM documents")
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Error listing documents: {e}")
            raise

    def delete_document(self, document_id):
        """
        Delete the document metadata from SQLite.

        Args:
            document_id (str): Document ID.
        """
        try:
            self.cursor.execute("DELETE FROM documents WHERE id=?", (document_id,))
            self.conn.commit()
            logging.info(f"Document {document_id} deleted from database.")
        except sqlite3.Error as e:
            logging.error(f"Error deleting document {document_id}: {e}")
            raise

    def close(self):
        """
        Close the SQLite database connection and cursor.
        """
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            logging.info("Database connection closed.")
        except sqlite3.Error as e:
            logging.error(f"Error closing database connection: {e}")
            raise

    def query_embeddings(self, query_vector, top_k=5):
        """
        Query the FAISS index for the most similar embeddings.

        Args:
            query_vector (np.ndarray): The query embedding vector.
            top_k (int): The number of top results to return.

        Returns:
            list: List of indices of the top_k similar embeddings.
        """
        try:
            query_vector = np.array(query_vector).astype(np.float32).reshape(1, -1)
            distances, indices = self.index.search(query_vector, top_k)
            logging.info(f"Queried FAISS index with top_k={top_k}")
            return indices[0]
        except Exception as e:
            logging.error(f"Error querying embeddings: {e}")
            raise

    def get_document_metadata(self, document_ids):
        """
        Retrieve the metadata for the given document IDs.

        Args:
            document_ids (list): List of document IDs.

        Returns:
            list: List of tuples containing document IDs and file paths.
        """
        try:
            placeholders = ', '.join('?' for _ in document_ids)
            query = f"SELECT id, file_path FROM documents WHERE id IN ({placeholders})"
            self.cursor.execute(query, document_ids)
            logging.info(f"Retrieved metadata for document IDs: {document_ids}")
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Error retrieving metadata: {e}")
            raise

    def index_to_id(self, index):
        """
        Convert an index from the FAISS index to a document ID.

        Args:
            index (int): Index from the FAISS index.

        Returns:
            str: Corresponding document ID.
        """
        return self.index_to_id_map.get(index)

    def save_mappings(self):
        """
        Save the index-to-ID mappings to a file.
        """
        try:
            with open(self.sqlite_db_path + '_mappings.pkl', 'wb') as f:
                pickle.dump((self.id_to_index, self.index_to_id_map), f)
            logging.info("Saved index-to-ID mappings.")
        except Exception as e:
            logging.error(f"Error saving mappings: {e}")
            raise

    def load_mappings(self):
        """
        Load the index-to-ID mappings from a file.
        """
        try:
            if os.path.exists(self.sqlite_db_path + '_mappings.pkl'):
                with open(self.sqlite_db_path + '_mappings.pkl', 'rb') as f:
                    self.id_to_index, self.index_to_id_map = pickle.load(f)
                logging.info("Loaded index-to-ID mappings.")
        except Exception as e:
            logging.error(f"Error loading mappings: {e}")
            raise
