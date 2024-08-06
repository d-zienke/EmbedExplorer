# EmbedExplorer

EmbedExplorer is a Python application for processing text documents, generating embeddings using the all-MiniLM-L6-v2 model from Sentence Transformers, and storing them in a local vector database. The project uses SQLite for metadata storage and FAISS for vector storage. The entire process runs locally and offline for document processing, but it uses the OpenAI GPT model for real-time chatbot responses.

## Features

- Extracts text from PDF, TXT, and Markdown files.
- Chunks text into smaller segments.
- Generates embeddings using the all-MiniLM-L6-v2 model.
- Stores embeddings in a local FAISS vector database.
- Manages document metadata in a local SQLite database.
- Supports CRUD operations on the document metadata.
- Provides a real-time chatbot that can answer queries based on the embedded documents using OpenAI GPT.

## Project Structure

```txt
EmbedExplorer/
│
├── vector_db/
│   ├── __init__.py
│   ├── database.py
│   └── document_processor.py
│
├── chatbot/
│   ├── __init__.py
│   ├── model_handler.py
│   ├── query_handler.py
│   └── response_generator.py
│
├── knowledge/
│   └── text_documents/ # Place your text documents here
│
├── database/ # Local database files
│
├── tests/
│   └── test_database.py # Unit tests
│
├── config.py # Global configuration
├── main.py # Main entry point for document processing
├── start_chatbot.py # Integrated chatbot and query example
├── .env # Environment variables for secrets
├── venv # Virtual environment
└── requirements.txt # Dependencies
```

## Installation

1. **Clone the repository**:

   ```sh
   git clone https://github.com/d-zienke/EmbedExplorer.git
   cd EmbedExplorer
   ```

2. **Set up a virtual environment**:

   ```sh
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

4. **Configure the Environment**:

Create an `.env` file in the root directory and add your Hugging Face token and/or OpenAI token.

_(If you don't intend to use OpenAI's model, write anything for OPENAI_API_KEY)_

```env
HUGGINGFACE_TOKEN="your_huggingface_token_here"
OPENAI_API_KEY="your_openai_token_here"
```

## Configuration

Edit the `config.py` file to configure the chunk size, overlap size, paths for the SQLite and FAISS databases, and model settings

```py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # General settings
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SQLITE_DB_PATH = "database/metadata.db"
    FAISS_INDEX_PATH = "database/faiss.index"
    EMBEDDING_DIMENSION = 384  # Dimension of the embeddings used

    # Chatbot settings
    MODEL_TYPE = "gpt-4o-mini"
    LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
    GPT4_MODEL_NAME = "gpt-4o-mini"
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
        "from the documents, state that explicitly. If a query requests general information or opinions, make "
        "it clear that your primary function is to provide information based on the embedded documents."
    )

```

## Usage

1. Place your text documents in the `knowledge/text_documents/` directory.
2. Run the main application:

   ```sh
   python main.py
   ```

   The application will automatically create the necessary directories (`database` and `knowledge/text_documents`) if they don't exist.

3. Run the chatbot:

   ```sh
   python start_chatbot.py --mode chatbot
   ```

4. Test the query mechanism:

   ```sh
   python start_chatbot.py --mode test
   ```

## Running Tests

To run the unit tests, execute:

```sh
python -m unittest discover -s tests
```

## Code Overview

### `vector_db/database.py`

Manages the SQLite and FAISS database operations.

### `vector_db/document_processor.py`

Processes documents, extracts text, chunks text, generates embeddings, and stores them in the database.

### `chatbot/model_handler.py`

Manages the language models (GPT-4o-mini and LLaMA), generates embeddings using SBERT, generates responses using the appropriate model based on configuration.

### `chatbot/query_handler.py`

Handles document retrieval based on query embeddings.

### `chatbot/response_generator.py`

Generates chatbot responses using the LLaMA model and OpenAI GPT-4o-mini model.

### `tests/test_database.py`

Contains unit tests for the VectorDatabase class.

### `main.py`

The main entry point of the application. Processes all documents in the knowledge/text_documents/ directory.

### `start_chatbot.py`

Integrates the chatbot functionality and provides a mechanism to test the query.

## Contributing

Feel free to submit issues, fork the repository, and send pull requests. For major changes, please open an issue first to discuss what you would like to change.

The `main` branch is locked and read-only. Please always create a new branch for your changes.

Keep your branch names and commit messages clear, consistent, and easy to understand. Please adhere to the following naming convention.

### Branch Naming Convention

1. Feature Branches:

- Use the prefix **feature\_** followed by a brief description of the feature with hyphens separating words.
- Example: **feature_user-authentication**

2. Bugfix Branches:

- Use the prefix **bugfix\_** followed by a brief description of the bug with hyphens separating words.
- Example: **bugfix_fix-login-error**

3. Hotfix Branches:

- Use the prefix **hotfix\_** for urgent fixes in production with hyphens separating words.
- Example: **hotfix_security-patch**

4. Release Branches:

- Use the prefix **release\_** followed by the version number.
- Example: **release_v1.2.0**

5. Experimental Branches:

- Use the prefix **exp\_** for experimental features or spikes with hyphens separating words.
- Example: **exp_new-ui-experiment**

6. Documentation Branches:

- Use the prefix **docs\_** followed by a brief description of the documentation update with hyphens separating words.
- Example: **docs_update-readme**

### Commit Message Convention

1. Type:

- **feat**: A new feature.
- **fix**: A bug fix.
- **docs**: Documentation changes.
- **style**: Code style changes (formatting, missing semi-colons, etc).
- **refactor**: Code refactoring without changing functionality.
- **perf**: Performance improvements.
- **test**: Adding or updating tests.
- **chore**: Other changes that don't modify src or test files.

2. Description:

- Keep the first line (summary) under 50 characters.
- Use the imperative mood (e.g., "Add", "Fix", "Update").

3. Body (optional):

- Use if a more detailed explanation is necessary.
- Separate from the summary with a blank line.
- Explain the motivation for the change and contrast with the previous behavior.

### Examples:

#### Branch Names:

- feature_user-authentication
- bugfix_fix-login-error
- hotfix_security-patch
- release_v1.2.0
- exp_new-ui-experiment

#### Commit Messages:

- feat: Add user authentication
- fix: Correct login error when user is inactive
- docs: Update API documentation
- style: Format code according to new guidelines
- refactor: Reorganize user model
- perf: Improve query performance for dashboard
- test: Add unit tests for login functionality
- chore: Update dependencies

## License

This project is licensed under the Apache License Version 2.0. See the LICENSE file for details.
