import logging
from concurrent.futures import ThreadPoolExecutor
from chatbot.query_handler import QueryHandler
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class ResponseGenerator:
    def __init__(self):
        self.query_handler = QueryHandler()

    def generate_responses(self, prompts, system_prompt=None):
        """
        Generate responses using the GPT-4o model in parallel.
        Args:
            prompts (list): List of prompts to generate responses for.
            system_prompt (str): System prompt to guide the model.
        Returns:
            list: List of generated responses.
        """
        with ThreadPoolExecutor() as executor:
            responses = list(
                executor.map(lambda prompt: self.query_handler.model_handler.generate_response(prompt, system_prompt), prompts)
            )
        return responses

    def generate_chatbot_response(self, query_text, metadata):
        """
        Generate a chatbot response based on the query text and document metadata.
        Args:
            query_text (str): The query text.
            metadata (list): List of document metadata.
        Returns:
            str: Formatted chatbot response.
        """
        file_paths = [file_path for _, _, file_path in metadata]
        contents = self.query_handler.read_documents_in_parallel(file_paths)

        prompts = [
            f"Based on the following content:\n\n{content[:500]}\n\nAnswer the question: {query_text}"
            for content in contents
        ]

        responses = self.generate_responses(prompts, Config.SYSTEM_PROMPT)

        # Filter responses to only include those with relevant information
        filtered_responses = [
            (doc_id, file_path, response)
            for (doc_id, _, file_path), response in zip(metadata, responses)
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

    def decide_response(self, query_text):
        """
        Decide whether to query the database or generate a natural conversation response.
        Args:
            query_text (str): The query text.
        Returns:
            str: The generated response.
        """
        # Normalize the query text for easier comparison
        query_text_lower = query_text.lower().strip()

        # Check if the user is asking to list document titles
        if query_text_lower in ["list documents", "show documents", "what documents do you have", "list all document titles"]:
            titles = self.query_handler.list_document_titles()
            if not titles:
                return "No documents found in the database."
            return "Here are the available documents:\n" + "\n".join(titles)

        # Existing logic for content-based queries
        keywords = ["information", "details", "document", "explain", "describe"]

        if any(keyword in query_text_lower for keyword in keywords):
            metadata = self.query_handler.retrieve_relevant_documents(query_text)
            response = self.generate_chatbot_response(query_text, metadata)
        else:
            response = self.query_handler.model_handler.generate_response(query_text)

        return response
