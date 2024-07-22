from concurrent.futures import ThreadPoolExecutor
import logging
from dotenv import load_dotenv
import os
import json
from chatbot.model_handler import ModelHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime=s - %(levelname=s - %(message=s')

# Load the environment variables from the .env file
load_dotenv()

# Load configuration from chatbot/config.json
with open('chatbot/config.json', 'r') as file:
    config = json.load(file)

# Initialize the model handler
model_handler = ModelHandler(config)

def generate_llama_responses(prompts, system_prompt=None):
    with ThreadPoolExecutor() as executor:
        responses = list(executor.map(lambda prompt: model_handler.generate_response(prompt, system_prompt), prompts))
    return responses

def generate_chatbot_response(query_text, metadata):
    from chatbot.query_handler import read_documents_in_parallel
    
    file_paths = [file_path for _, file_path in metadata]
    contents = read_documents_in_parallel(file_paths)
    
    prompts = [
        f"Based on the following content:\n\n{content[:500]}\n\nAnswer the question: {query_text}"
        for content in contents
    ]

    responses = generate_llama_responses(prompts, config.get('system_prompt'))

    formatted_responses = [
        f"Document ID: {doc_id}\nFile Path: {file_path}\nResponse:\n{response}\n"
        for (doc_id, file_path), response in zip(metadata, responses)
    ]

    return "\n\n".join(formatted_responses)
