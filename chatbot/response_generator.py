from concurrent.futures import ThreadPoolExecutor
import logging
from dotenv import load_dotenv
import os
import json
from chatbot.model_handler import ModelHandler
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the environment variables from the .env file
load_dotenv()

# Initialize the model handler
model_handler = ModelHandler()


def generate_llama_responses(prompts, system_prompt=None):
    """
    Generate responses using the Llama model in parallel.

    Args:
        prompts (list): List of prompts to generate responses for.
        system_prompt (str): System prompt to guide the model.

    Returns:
        list: List of generated responses.
    """
    with ThreadPoolExecutor() as executor:
        responses = list(executor.map(lambda prompt: model_handler.generate_response(prompt, system_prompt), prompts))
    return responses


def generate_chatbot_response(query_text, metadata):
    """
    Generate a chatbot response based on the query text and document metadata.

    Args:
        query_text (str): The query text.
        metadata (list): List of document metadata.

    Returns:
        str: Formatted chatbot response.
    """
    from chatbot.query_handler import read_documents_in_parallel

    file_paths = [file_path for _, file_path in metadata]
    contents = read_documents_in_parallel(file_paths)

    prompts = [
        f"Based on the following content:\n\n{content[:500]}\n\nAnswer the question: {query_text}"
        for content in contents
    ]

    responses = generate_llama_responses(prompts, Config.SYSTEM_PROMPT)

    # Filter responses to only include those with relevant information
    filtered_responses = [
        (doc_id, file_path, response)
        for (doc_id, file_path), response in zip(metadata, responses)
        if "does not provide information" not in response
    ]

    # Check if no relevant data was found in any document
    if not filtered_responses:
        return "No relevant information was found in any of the documents."

    # Format the filtered responses
    formatted_responses = [
        f"Document ID: {doc_id}\nFile Path: {file_path}\nResponse:\n{response}\n"
        for doc_id, file_path, response in filtered_responses
    ]

    return "\n\n".join(formatted_responses)
