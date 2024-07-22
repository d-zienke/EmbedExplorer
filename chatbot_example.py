from chatbot.query_handler import retrieve_relevant_documents
from chatbot.response_generator import generate_chatbot_response

# Example usage in a chatbot
query_text = "What is LLM? in one sentence"
metadata = retrieve_relevant_documents(query_text)
response = generate_chatbot_response(query_text, metadata)
print(response)
