import json
import os
from vector_db.database import VectorDatabase
from vector_db.document_processor import DocumentProcessor

def main():
    # Load configuration
    with open('vector_db/config.json', 'r') as file:
        config = json.load(file)

    # Initialize vector database and document processor
    vector_db = VectorDatabase(config)
    doc_processor = DocumentProcessor(config, vector_db)

    # Directory containing documents
    knowledge_path = 'knowledge/text_documents'

    # Process each document in the directory
    for file_name in os.listdir(knowledge_path):
        file_path = os.path.join(knowledge_path, file_name)
        if os.path.isfile(file_path):
            doc_processor.process_document(file_path)

    # List stored documents
    documents = vector_db.list_documents()
    print("Stored documents:", documents)

    # Close the vector database
    vector_db.close()

if __name__ == "__main__":
    main()
