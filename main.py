import os
import argparse
import logging
from vector_db.database import VectorDatabase
from vector_db.document_processor import DocumentProcessor
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_directories():
    """
    Ensure necessary directories exist.
    """
    os.makedirs('database', exist_ok=True)
    os.makedirs('knowledge/text_documents', exist_ok=True)
    logging.info("Necessary directories ensured.")

def main():
    """
    Main entry point for the application.
    """
    parser = argparse.ArgumentParser(description="EmbedExplorer")
    parser.add_argument('--log', type=str, default='info', help='Logging level (debug, info, warning, error, critical)')
    args = parser.parse_args()

    logging_level = getattr(logging, args.log.upper(), logging.INFO)
    logging.getLogger().setLevel(logging_level)

    try:
        ensure_directories()  # Ensure necessary directories exist

        vector_db = VectorDatabase()
        doc_processor = DocumentProcessor(vector_db)

        knowledge_path = 'knowledge/text_documents'
        for file_name in os.listdir(knowledge_path):
            file_path = os.path.join(knowledge_path, file_name)
            if os.path.isfile(file_path):
                try:
                    doc_processor.process_document(file_path)
                except Exception as e:
                    logging.error(f"Error processing document {file_path}: {e}")

        logging.info("Stored documents: %s", vector_db.list_documents())
    except Exception as e:
        logging.critical(f"Critical error in main execution: {e}")
    finally:
        vector_db.close()

if __name__ == "__main__":
    main()
