import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from chatbot.query_handler import QueryHandler
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class ResponseGenerator:
    def __init__(self, context_window_size=5):
        self.query_handler = QueryHandler()
        self.context_window_size = context_window_size
        self.session_id = str(uuid.uuid4())  # Unique identifier for the session

    def add_to_context(self, user_prompt, chatbot_response):
        """
        Add a user prompt and corresponding chatbot response to the context.
        Also stores the interaction in the database.
        """
        self.query_handler.vector_db.add_conversation_entry(
            self.session_id, user_prompt, chatbot_response
        )

    def generate_responses(self, prompts, system_prompt=None):
        """
        Generate responses using the GPT-4o model, incorporating context.
        """
        # Retrieve recent conversation context
        recent_conversation = self.query_handler.vector_db.get_recent_conversation(
            self.session_id, limit=self.context_window_size
        )
        context_str = "\n".join([f"User: {user}\nBot: {bot}" for user, bot in recent_conversation])

        # Integrate context into the prompt
        full_prompt = f"{context_str}\nUser: {prompts[-1]}"

        response = self.query_handler.model_handler.generate_response(full_prompt, system_prompt)
        self.query_handler.vector_db.add_conversation_entry(self.session_id, prompts[-1], response)
        return [response]

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
        Decide the response based on recognized intent or default behavior.
        """
        action_response = self.query_handler.decide_action(query_text)
        if action_response:
            return action_response

        # Existing logic for content-based queries
        metadata = self.query_handler.retrieve_relevant_documents(query_text)
        response = self.generate_chatbot_response(query_text, metadata)

        # Add the current interaction to the context
        self.add_to_context(query_text, response)

        return response

    def generate_parallel_responses(self, prompts, system_prompt=None):
        """
        Generate responses in parallel for multiple prompts.
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

    def reset_context(self):
        """
        Reset the conversation context for the current session.
        """
        self.query_handler.vector_db.clear_conversation(self.session_id)
        return "Conversation context has been reset."

    def list_documents(self):
        """
        List all available documents in the vector database.
        Returns:
            str: Formatted list of document titles.
        """
        titles = self.query_handler.list_document_titles()
        if not titles:
            return "No documents found in the database."
        return "Here are the available documents:\n" + "\n".join(titles)

    def retrieve_context(self):
        """
        Retrieve the current conversation context for the session.
        Returns:
            str: The context as a formatted string.
        """
        recent_conversation = self.query_handler.vector_db.get_recent_conversation(
            self.session_id, limit=self.context_window_size
        )
        return "\n".join([f"User: {user}\nBot: {bot}" for user, bot in recent_conversation])
