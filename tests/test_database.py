import os
import unittest
import numpy as np
import shutil
from vector_db.database import VectorDatabase


class TestVectorDatabase(unittest.TestCase):
    """
    Unit tests for the VectorDatabase class.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment before running any tests.
        """
        cls.test_dir = 'test_database'
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.config = {
            "CHUNK_SIZE": 300,
            "CHUNK_OVERLAP": 50,
            "SQLITE_DB_PATH": os.path.join(cls.test_dir, "metadata.db"),
            "FAISS_INDEX_PATH": os.path.join(cls.test_dir, "faiss.index"),
            "EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "EMBEDDING_DIMENSION": 384  # Ensure this matches the embedding model's dimension
        }

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the test environment after all tests have run.
        """
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        """
        Set up the test environment before each test.
        """
        self.vector_db = VectorDatabase(self.config)
        self.vector_db.clear_database()  # Clear the database before each test

    def tearDown(self):
        """
        Clean up the test environment after each test.
        """
        self.vector_db.close()
        if os.path.exists(self.config["SQLITE_DB_PATH"]):
            os.remove(self.config["SQLITE_DB_PATH"])
        if os.path.exists(self.config["FAISS_INDEX_PATH"]):
            os.remove(self.config["FAISS_INDEX_PATH"])

    def test_create_and_read_document(self):
        """
        Test creating and reading a document in the database.
        """
        document_id = 'test_doc_1'
        file_path = 'test_path_1'
        embeddings = [np.random.rand(self.config["EMBEDDING_DIMENSION"]).astype(np.float32)]

        self.vector_db.mark_document_as_processed(document_id, file_path)
        self.vector_db.store_embeddings(document_id, embeddings)

        self.assertTrue(self.vector_db.is_document_processed(document_id))

        documents = self.vector_db.list_documents()
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0][0], document_id)
        self.assertEqual(documents[0][1], file_path)

    def test_delete_document(self):
        """
        Test deleting a document from the database.
        """
        document_id = 'test_doc_2'
        file_path = 'test_path_2'
        embeddings = [np.random.rand(self.config["EMBEDDING_DIMENSION"]).astype(np.float32)]

        self.vector_db.mark_document_as_processed(document_id, file_path)
        self.vector_db.store_embeddings(document_id, embeddings)

        self.vector_db.delete_document(document_id)

        self.assertFalse(self.vector_db.is_document_processed(document_id))

        documents = self.vector_db.list_documents()
        self.assertEqual(len(documents), 0)


if __name__ == '__main__':
    unittest.main()
