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
