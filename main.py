import os
import json
import argparse
from vector_db.database import VectorDatabase
from vector_db.document_processor import DocumentProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    """
    Load the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = json.load(file)
    
    # Validate config
    required_keys = ["chunk_size", "chunk_overlap", "sqlite_db_path", "faiss_index_path", "embedding_model"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    return config

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
    parser.add_argument('--config', type=str, default='vector_db/config.json', help='Path to the configuration file')
    parser.add_argument('--log', type=str, default='info', help='Logging level (debug, info, warning, error, critical)')
    args = parser.parse_args()

    logging_level = getattr(logging, args.log.upper(), logging.INFO)
    logging.getLogger().setLevel(logging_level)

    try:
        ensure_directories()  # Ensure necessary directories exist

        config = load_config(args.config)
        vector_db = VectorDatabase(config)
        doc_processor = DocumentProcessor(config, vector_db)

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
