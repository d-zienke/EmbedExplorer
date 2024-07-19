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

## License
This project is licensed under the MIT License. See the LICENSE file for details.