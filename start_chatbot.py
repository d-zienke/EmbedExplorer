import logging
from chatbot.response_generator import ResponseGenerator
from vector_db.database import VectorDatabase
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_query():
    # Initialize the vector database
    vector_db = VectorDatabase()

    # Example query text
    query_text = "little golden key in the lock"

    # Generate the query embedding
    model_handler = ResponseGenerator().query_handler.model_handler
    query_vector = model_handler.generate_embedding(query_text)
    logging.info(f"Query vector generated.")

    # Perform the query
    top_indices = vector_db.query_embeddings(query_vector, top_k=3)
    logging.info(f"Top indices from FAISS query: {top_indices}")

    # Retrieve metadata for the top results
    document_ids = [vector_db.index_to_id(idx) for idx in top_indices]
    logging.info(f"Document IDs retrieved: {document_ids}")

    metadata = vector_db.get_document_metadata(document_ids)
    logging.info(f"Metadata retrieved: {metadata}")

    # Print the results
    for doc_id, file_path in metadata:
        print(f"Document ID: {doc_id}, File Path: {file_path}")

    # Close the vector database
    vector_db.close()

def chatbot():
    response_generator = ResponseGenerator()
    print("Chatbot is ready to converse! Type your query below.")
    while True:
        query_text = input("You: ")
        if query_text.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        response = response_generator.decide_response(query_text)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EmbedExplorer Chatbot")
    parser.add_argument('--mode', type=str, choices=['chatbot', 'test'], default='chatbot', help='Run mode: chatbot or test')
    args = parser.parse_args()

    if args.mode == 'test':
        test_query()
    else:
        chatbot()
