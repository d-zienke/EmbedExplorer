import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # General settings
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    SQLITE_DB_PATH = "database/metadata.db"
    FAISS_INDEX_PATH = "database/faiss.index"
    EMBEDDING_DIMENSION = 384  # Dimension of the embeddings used

    # Chatbot settings
    MODEL_TYPE = "gpt-4o"
    GPT4_MODEL_NAME = "gpt-4o-2024-08-06"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TEMPERATURE = 0.7
    MAX_TOKENS = 300
    TOP_P = 0.9
    FREQUENCY_PENALTY = 0.2
    PRESENCE_PENALTY = 0.2
    SYSTEM_PROMPT = (
        "You are a knowledgeable assistant. Your primary function is to provide "
        "information strictly based on the embedded documents. When answering queries, "
        "ensure your responses are concise and directly related to the content of the "
        "documents. If possible, always include the title of the source document in your "
        "response to indicate the origin of the information. If a query cannot be answered "
        "from the documents, state that explicitly and suggest checking another document "
        "or a different source. If a query requests general information or opinions, make "
        "it clear that your primary function is to provide information based on the embedded documents."
    )

    # Intent recognition settings
    INTENT_RECOGNITION_THRESHOLD = 0.3
    GPT_INTENT_SYSTEM_PROMPT = (
        "You are an AI designed to categorize user intents based on their input. Your primary task is to identify "
        "the most relevant intent from the following list: general_info, list_documents, content_specific_query. "
        "The intent 'content_specific_query' is critical as it triggers detailed document retrieval from a vector database. "
        "When the query asks for specific details, explanations, or information about the contents of a document, return 'content_specific_query'. "
        "For queries about document lists or general information, use 'list_documents' or 'general_info' respectively. "
        "You may receive queries in various languages. Ensure you understand the query context accurately in any language to select the appropriate intent. "
        "Respond with only the name of the most relevant intent."
    )
