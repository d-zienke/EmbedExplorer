import os
import sqlite3
import faiss
import numpy as np

class VectorDatabase:
    """
    Class for managing the SQLite and FAISS database operations.
    """

    def __init__(self, config):
        """
        Initialize the VectorDatabase class.

        Args:
            config (dict): Configuration dictionary.
        """
        self.config = config
        self.dimension = 1024  # Ensure this matches the dimension of your embedding vectors
        self.conn = None
        self.cursor = None
        self.setup_sqlite()
        self.setup_faiss()

    def setup_sqlite(self):
        """
        Set up the SQLite database connection and cursor.
        """
        self.conn = sqlite3.connect(self.config["sqlite_db_path"])
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                file_path TEXT,
                status TEXT
            )
        ''')
        self.conn.commit()
        print(f"\nSQLite setup complete.")

    def setup_faiss(self):
        """
        Set up the FAISS index.
        """
        if os.path.exists(self.config["faiss_index_path"]):
            self.index = faiss.read_index(self.config["faiss_index_path"])
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
        print(f"\nFAISS setup complete.")

    def clear_database(self):
        """
        Clear the SQLite and FAISS databases.
        """
        self.cursor.execute("DELETE FROM documents")
        self.conn.commit()
        self.index.reset()
        faiss.write_index(self.index, self.config["faiss_index_path"])
        print(f"\nDatabase cleared.")

    def store_embeddings(self, document_id, embeddings):
        """
        Store the embeddings in the FAISS index.

        Args:
            document_id (str): Document ID.
            embeddings (list): List of embeddings.
        """
        embeddings_array = np.array(embeddings).astype(np.float32)
        if embeddings_array.ndim == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
        assert embeddings_array.shape[1] == self.dimension, f"Embedding dimension mismatch: {embeddings_array.shape[1]} != {self.dimension}"
        self.index.add(embeddings_array)
        faiss.write_index(self.index, self.config["faiss_index_path"])
        print(f"Embeddings for document {document_id} stored.")

    def is_document_processed(self, document_id):
        """
        Check if the document is already processed.

        Args:
            document_id (str): Document ID.

        Returns:
            bool: True if the document is processed, False otherwise.
        """
        self.cursor.execute("SELECT status FROM documents WHERE id=?", (document_id,))
        result = self.cursor.fetchone()
        return result is not None

    def mark_document_as_processed(self, document_id, file_path):
        """
        Mark the document as processed.

        Args:
            document_id (str): Document ID.
            file_path (str): File path of the document.
        """
        if not self.is_document_processed(document_id):
            self.cursor.execute("INSERT INTO documents (id, file_path, status) VALUES (?, ?, ?)", (document_id, file_path, 'processed'))
            self.conn.commit()
            print(f"Document {document_id} marked as processed.")

    def list_documents(self):
        """
        List all processed documents.

        Returns:
            list: List of tuples containing document IDs and file paths.
        """
        self.cursor.execute("SELECT id, file_path FROM documents")
        return self.cursor.fetchall()

    def delete_document(self, document_id):
        """
        Delete the document metadata from SQLite.

        Args:
            document_id (str): Document ID.
        """
        self.cursor.execute("DELETE FROM documents WHERE id=?", (document_id,))
        self.conn.commit()
        print(f"Document {document_id} deleted from database.")

    def close(self):
        """
        Close the SQLite database connection and cursor.
        """
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print(f"\nDatabase connection closed.")
