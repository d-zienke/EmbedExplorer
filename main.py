import os
import json
import argparse
from vector_db.database import VectorDatabase
from vector_db.document_processor import DocumentProcessor

def load_config(config_path):
    """
    Load the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        return json.load(file)

def ensure_directories():
    """
    Ensure necessary directories exist.
    """
    os.makedirs('database', exist_ok=True)
    os.makedirs('knowledge/text_documents', exist_ok=True)
    print("Necessary directories ensured.")

def main():
    """
    Main entry point for the application.
    """
    parser = argparse.ArgumentParser(description="EmbedExplorer")
    parser.add_argument('--config', type=str, default='vector_db/config.json', help='Path to the configuration file')
    args = parser.parse_args()

    ensure_directories()  # Ensure necessary directories exist

    config = load_config(args.config)
    vector_db = VectorDatabase(config)
    doc_processor = DocumentProcessor(config, vector_db)

    knowledge_path = 'knowledge/text_documents'
    for file_name in os.listdir(knowledge_path):
        file_path = os.path.join(knowledge_path, file_name)
        if os.path.isfile(file_path):
            doc_processor.process_document(file_path)

    print("Stored documents:", vector_db.list_documents())
    vector_db.close()

if __name__ == "__main__":
    main()
