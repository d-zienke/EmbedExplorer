from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
import os
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

        if self.model_type == 'gpt-4o-mini':
            openai.api_key = Config.OPENAI_API_KEY
            self.client = openai.OpenAI(api_key=openai.api_key)
        elif self.model_type == 'llama':
            self.tokenizer = AutoTokenizer.from_pretrained(Config.LLAMA_MODEL_NAME,
                                                           token=os.getenv('HUGGINGFACE_TOKEN'))
            self.model = AutoModelForCausalLM.from_pretrained(Config.LLAMA_MODEL_NAME,
                                                              token=os.getenv('HUGGINGFACE_TOKEN'))
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
        if self.model_type == 'gpt-4o-mini':
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
            return response.choices[0].message.content
        elif self.model_type == 'llama':
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
