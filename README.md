# EmbedExplorer

EmbedExplorer is a Python application for processing text documents, generating embeddings using the Ollama embedding model, and storing them in a local vector database. The project uses SQLite for metadata storage and FAISS for vector storage. The entire process runs locally and offline.

## Features

- Extracts text from PDF, TXT, and Markdown files.
- Chunks text into smaller segments.
- Generates embeddings using the `mxbai-embed-large` model from Ollama.
- Stores embeddings in a local FAISS vector database.
- Manages document metadata in a local SQLite database.
- Supports CRUD operations on the document metadata.

## Project Structure

```txt
EmbedExplorer/
│
├── vector_db/
│ ├── init.py
│ ├── config.json
│ ├── database.py
│ └── document_processor.py
│
├── knowledge/
│ └── text_documents/ # Place your text documents here
│
├── database/ # Local database files
│
├── tests/
│ └── test_database.py # Unit tests
│
├── main.py # Main entry point
│
└── requirements.txt # Dependencies
```

## Installation

1. **Clone the repository**:

   ```sh
   git clone <repository_url>
   cd EmbedExplorer
   ```

2. **Set up a virtual environment**:

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

4. **Download the Ollama Embedding Model**:

   ```sh
   ollama pull mxbai-embed-large
   ```

## Configuration

Edit the `config.json` file in the `vector_db/` directory to configure chunk size, overlap size, and paths for the SQLite and FAISS databases.

```json
{
	"chunk_size": 300,
	"chunk_overlap": 50,
	"sqlite_db_path": "database/metadata.db",
	"faiss_index_path": "database/faiss.index",
	"embedding_model": "mxbai-embed-large"
}
```

## Usage

1. Place your text documents in the knowledge/text_documents/ directory.
2. Run the main application:

```sh
python main.py
```

The application will automatically create the necessary directories (database and knowledge/text_documents) if they don't exist.

3. View stored documents:

The processed documents and their embeddings will be stored in the local database.

## Running Tests

To run the unit tests, execute:

```sh
python -m unittest discover -s tests
```

## Code Overview

### vector_db/database.py

Manages the SQLite and FAISS database operations.

### vector_db/document_processor.py

Processes documents, extracts text, chunks text, generates embeddings, and stores them in the database.

### tests/test_database.py

Contains unit tests for the VectorDatabase class.

### main.py

The main entry point of the application. Processes all documents in the knowledge/text_documents/ directory.

## Contributing

Feel free to submit issues, fork the repository, and send pull requests. For major changes, please open an issue first to discuss what you would like to change.

Keep your branch names and commit messages clear, consistent, and easy to understand. Please adhere to the following naming convention.

### Branch Naming Convention

1. Feature Branches:

- Use the prefix feature\_ followed by a brief description of the feature with hyphens separating words.
- Example: feature_user-authentication

2. Bugfix Branches:

- Use the prefix bugfix\_ followed by a brief description of the bug with hyphens separating words.
- Example: bugfix_fix-login-error

3. Hotfix Branches:

- Use the prefix hotfix\_ for urgent fixes in production with hyphens separating words.
- Example: hotfix_security-patch

4. Release Branches:

- Use the prefix release\_ followed by the version number.
- Example: release_v1.2.0

5. Experimental Branches:

- Use the prefix exp\_ for experimental features or spikes with hyphens separating words.
- Example: exp_new-ui-experiment

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

- feat: add user authentication
- fix: correct login error when user is inactive
- docs: update API documentation
- style: format code according to new guidelines
- refactor: reorganize user model
- perf: improve query performance for dashboard
- test: add unit tests for login functionality
- chore: update dependencies

## License

This project is licensed under the Apache License Version 2.0. See the LICENSE file for details.
