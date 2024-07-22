# chatbot/response_generator.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from dotenv import load_dotenv
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message=s')

# Load the environment variables from the .env file
load_dotenv()

# Load configuration from chatbot/config.json
with open('chatbot/config.json', 'r') as file:
    config = json.load(file)

# Get the Hugging Face token from the environment variables
token = os.getenv('HUGGINGFACE_TOKEN')

# Check if the token is available
if token is None:
    raise ValueError("Hugging Face token not found in environment variables. Please set it in the .env file.")

# Function to download and cache the models
def cache_models():
    global llama_tokenizer, llama_model

    llama_model_name = config['llama_model_name']
    
    # Download and cache the tokenizer and model
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=token)
    llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name, token=token)
    logging.info(f"Models cached successfully.")

# Ensure models are cached
cache_models()

def generate_llama_response(prompt):
    """
    Generate a response using the LLaMA model.

    Args:
        prompt (str): The input prompt for the LLaMA model.

    Returns:
        str: The generated response.
    """
    inputs = llama_tokenizer(prompt, return_tensors="pt")
    outputs = llama_model.generate(**inputs)
    response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def generate_chatbot_response(query_text, metadata):
    """
    Generate a response for the chatbot based on the query text.

    Args:
        query_text (str): The user's query text.
        metadata (list): List of tuples containing document IDs and file paths.

    Returns:
        str: The chatbot's response.
    """
    from chatbot.query_handler import read_document
    
    responses = []

    for doc_id, file_path in metadata:
        content = read_document(file_path)
        snippet = content[:500]  # Limiting to 500 characters for brevity
        prompt = f"Based on the following content:\n\n{snippet}\n\nAnswer the question: {query_text}"
        response = generate_llama_response(prompt)
        responses.append(f"Document ID: {doc_id}\nFile Path: {file_path}\nResponse:\n{response}\n")

    return "\n\n".join(responses)
