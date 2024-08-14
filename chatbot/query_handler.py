import logging
import os
import re
import markdown
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import util
from chatbot.model_handler import ModelHandler
from vector_db.database import VectorDatabase
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class QueryHandler:
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.model_handler = ModelHandler()

    def recognize_intent(self, query_text):
        """
        Recognize the user's intent using a combination of GPT-based analysis and keyword matching.
        Args:
            query_text (str): The user's query text.
        Returns:
            str: Recognized intent (e.g., "list_documents", "reset_context").
        """
        # Check for immediate keywords or patterns
        immediate_intent = self.preliminary_intent_filter(query_text)
        if immediate_intent:
            return immediate_intent

        # Otherwise, use GPT-based analysis
        gpt_intent, confidence_score = self.model_handler.recognize_intent_with_gpt(query_text)
        if gpt_intent and confidence_score >= Config.INTENT_RECOGNITION_THRESHOLD:
            return gpt_intent

        # If GPT analysis is inconclusive, fallback to embedding-based similarity
        return self.embedding_based_intent_recognition(query_text)


    def preliminary_intent_filter(self, query_text):
        """
        Preliminary filter to check for specific keywords or patterns.
        Args:
            query_text (str): The user's query text.
        Returns:
            str or None: Recognized intent if a match is found, otherwise None.
        """
        keywords_to_intents = {
            "what are you": "general_info",
            "who are you": "general_info",
            "list documents": "list_documents",
            "show documents": "list_documents",
            # Add more immediate keyword checks as needed
        }

        query_lower = query_text.lower()
        for keyword, intent in keywords_to_intents.items():
            if keyword in query_lower:
                return intent
        return None

    def embedding_based_intent_recognition(self, query_text):
        """
        Recognize intent based on semantic similarity using embeddings.
        Args:
            query_text (str): The user's query text.
        Returns:
            str: Recognized intent.
        """
        intents = {
            "list_documents": [
                "list documents", "show documents", "what documents do you have", "list all document titles",
                "display the documents you have", "what files are in the database"
            ],
            "general_info": [
                "what are you", "who are you", "what can you do", "tell me about yourself",
                "what is your function", "what services do you offer"
            ],
            "content_specific_query": [
                "explain this document", "give me details about", "describe the contents of",
                "tell me about the document", "what's in the file", "what is in this document"
            ],
        }

        query_embedding = self.model_handler.generate_embedding(query_text)
        best_intent = None
        highest_score = 0

        for intent, examples in intents.items():
            for example in examples:
                example_embedding = self.model_handler.generate_embedding(example)
                score = util.pytorch_cos_sim(query_embedding, example_embedding).item()

                if score > highest_score:
                    highest_score = score
                    best_intent = intent

        logging.info(f"Recognized intent: {best_intent} with a score of {highest_score}")
        return best_intent

    def decide_action(self, query_text):
        """
        Decide the appropriate action based on the recognized intent and context.
        Args:
            query_text (str): The user's query text.
        Returns:
            str: Action to be performed by the chatbot.
        """
        intent = self.recognize_intent(query_text)
        if intent == "list_documents":
            titles = self.list_document_titles()
            if not titles:
                return "No documents found in the database."
            return "Here are the available documents:\n" + "\n".join(titles)

        if intent == "general_info":
            return "I am a document-based assistant designed to provide information strictly based on the embedded documents."

        if intent == "content_specific_query":
            metadata = self.retrieve_relevant_documents(query_text)
            return self.generate_chatbot_response(query_text, metadata)

        # Default action if no specific intent is recognized
        return None

    def retrieve_relevant_documents(self, query_text, top_k=3):
        """
        Retrieve the most relevant documents for the given query text.
        Args:
            query_text (str): Query text to search for.
            top_k (int): Number of top relevant documents to retrieve.
        Returns:
            list: Metadata of the top relevant documents.
        """
        query_vector = self.model_handler.generate_embedding(query_text)
        logging.info("Query vector generated.")
        top_indices = self.vector_db.query_embeddings(query_vector, top_k=top_k)
        logging.info(f"Top indices from FAISS query: {top_indices}")
        document_ids = [self.vector_db.index_to_id(idx) for idx in top_indices]
        logging.info(f"Document IDs retrieved: {document_ids}")
        metadata = self.vector_db.get_document_metadata(document_ids)
        logging.info(f"Metadata retrieved: {metadata}")
        return metadata

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
        contents = self.read_documents_in_parallel(file_paths)

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

    def generate_responses(self, prompts, system_prompt=None):
        """
        Generate responses using the GPT-4o model, incorporating context.
        Args:
            prompts (list): List of prompts to generate responses for.
            system_prompt (str): System prompt for additional context (optional).
        Returns:
            list: Generated responses.
        """
        with ThreadPoolExecutor() as executor:
            responses = list(
                executor.map(lambda prompt: self.model_handler.generate_response(prompt, system_prompt), prompts)
            )
        return responses

    def read_documents_in_parallel(self, file_paths):
        """
        Read multiple documents in parallel.
        Args:
            file_paths (list): List of file paths to read.
        Returns:
            list: List of text contents from the documents.
        """
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.read_document, file_paths))
        return results

    @staticmethod
    def read_document(file_path):
        """
        Read the contents of a document from the specified file path.
        Args:
            file_path (str): Path to the document file.
        Returns:
            str: Extracted text content from the document.
        """
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".pdf":
            return QueryHandler.extract_text_from_pdf(file_path)
        elif ext == ".txt":
            return QueryHandler.extract_text_from_txt(file_path)
        elif ext == ".md":
            return QueryHandler.extract_text_from_markdown(file_path)
        else:
            raise ValueError("Unsupported file type")

    @staticmethod
    def extract_text_from_pdf(file_path):
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = "".join([page.extract_text() for page in reader.pages])
        return text

    @staticmethod
    def extract_text_from_txt(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    @staticmethod
    def extract_text_from_markdown(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            md = file.read()
        html = markdown.markdown(md)
        return ''.join(re.findall(r'>[^<]+<', html))  # Extract text between tags

    def list_document_titles(self):
        """
        List all document titles stored in the database.
        Returns:
            list: List of document titles.
        """
        documents = self.vector_db.list_documents()
        titles = [doc[1] for doc in documents]
        logging.info(f"Retrieved document titles: {titles}")
        return titles
