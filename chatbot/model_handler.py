import openai
import os
import logging
from sentence_transformers import SentenceTransformer
from config import Config

class ModelHandler:
    """
    Handles model initialization and interactions for generating embeddings and responses.
    """
    def __init__(self):
        """
        Initialize the ModelHandler with the appropriate model based on the configuration.
        """
        self.model_type = Config.MODEL_TYPE
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)

        if self.model_type == 'gpt-4o':
            self.client = openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("Unsupported model type specified in config")

    def generate_embedding(self, text):
        """
        Generate an embedding for the given text using the configured embedding model.
        Args:
            text (str): Text to generate an embedding for.
        Returns:
            numpy.ndarray: Generated embedding.
        """
        return self.embedding_model.encode(text)

    def generate_response(self, prompt, system_prompt=None):
        """
        Generate a response to the given prompt using the configured language model.
        Args:
            prompt (str): User prompt for the model.
            system_prompt (str): System prompt for additional context (optional).
        Returns:
            str: Generated response.
        """
        try:
            if self.model_type == 'gpt-4o':
                messages = [
                    {"role": "system", "content": system_prompt or Config.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
                response = self.client.chat.completions.create(
                    model=Config.GPT4_MODEL_NAME,
                    messages=messages,
                    temperature=Config.TEMPERATURE,
                    max_tokens=Config.MAX_TOKENS,
                    top_p=Config.TOP_P,
                    frequency_penalty=Config.FREQUENCY_PENALTY,
                    presence_penalty=Config.PRESENCE_PENALTY
                )
                return response.choices[0].message['content']
        except Exception as e:
            logging.error(f"An error occurred while communicating with OpenAI: {e}")
            return "Sorry, I'm unable to generate a response at the moment. Please try again later."

    def recognize_intent_with_gpt(self, query_text):
        """
        Use GPT to recognize intent.
        Args:
            query_text (str): The user's query text.
        Returns:
            str: Recognized intent.
        """
        try:
            messages = [
                {"role": "system", "content": Config.GPT_INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": query_text}
            ]
            response = self.client.chat.completions.create(
                model=Config.GPT4_MODEL_NAME,
                messages=messages,
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
                top_p=Config.TOP_P,
                frequency_penalty=Config.FREQUENCY_PENALTY,
                presence_penalty=Config.PRESENCE_PENALTY
            )
            intent = response.choices[0].message['content'].strip().lower()
            logging.info(f"GPT recognized intent: {intent}")
            return intent, 1.0
        except Exception as e:
            logging.error(f"An error occurred while recognizing intent: {e}")
            return None, 0.0
