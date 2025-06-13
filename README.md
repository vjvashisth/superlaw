# GAL-RAG-System

A modular document parsing and semantic search pipeline using Unstructured.io, OpenAI embeddings, and FAISS — with a Streamlit interface for querying.

---

## 📦 Features

- Parse PDF, DOCX, PPTX, TXT, HTML, XLSX and CSV using [Unstructured.io](https://github.com/Unstructured-IO/unstructured)
- Chunk and normalize text content
- Generate embeddings using OpenAI `text-embedding-3-small`
- Store in a FAISS vector index for semantic retrieval
- Query through a clean Streamlit UI
- Fully Dockerized for deployment
- Includes examples and test suite

---

## 🧭 Project Structure

```bash
GAL-RAG-System/
├── config/                          # Configuration files (YAML)
│   └── config.yaml
├── examples/                        # Sample input and expected output formats
│   ├── sample_input.txt
│   └── sample_output.json
├── outputs/                         # Parsed + chunked output files (grouped by date)
│   └── 2025-05-18/
│       └── <filename>.json
├── samples/                         # Input documents (PDF, DOCX, PPTX, etc.)
│   └── example.pdf
├── src/                             # Core pipeline code
│   ├── app.py                       # Streamlit semantic search interface
│   ├── main.py                      # End-to-end document parsing + chunking
│   ├── embed_and_store.py           # OpenAI embedding + FAISS indexing
│   ├── parser_unstructured.py       # Parser using Unstructured.io
│   ├── parser_docetl.py             # Deprecated in favor of Unstructured-based pipeline
│   ├── chunking.py                  # Custom chunking logic
│   ├── file_utils.py                # File loader, MIME checker, folder setup
│   ├── metadata_extractor.py        # Metadata extractor from parsed output
│   └── schema.py                    # Output formatter + JSON writer
├── tests/                           # Unit tests for core modules
│   └── test_chunking.py
├── vector_index/                   # FAISS vector DB + metadata (generated)
│   ├── faiss.index
│   └── metadata.json
├── Dockerfile                       # Container setup for reproducible runs
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation

````

---

## 🚀 Quickstart

### 1. Clone the Repo

```bash
git clone https://github.com/GALBASE/GAL-RAG-System.git
cd GAL-RAG-System
```

### 2. Install Requirements (Locally)

```bash
pip install -r requirements.txt
```

### 3. Export Your OpenAI Key

```bash
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

---

## 🔄 Running the Pipeline

### ➤ Step 1: Parse and Chunk

```bash
python src/main.py
```

### ➤ Step 2: Generate Embeddings + Index

```bash
python src/embed_and_store.py
```

### ➤ Step 3: Launch the Streamlit App

```bash
streamlit run src/app.py
```

Then open: [http://localhost:8501](http://localhost:8501)

---

## 🐳 Run with Docker

### Build the Image

```bash
docker build -t gal-rag-system .
```

### Run the App with Mounted FAISS Index

```bash
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx \
  -v $PWD/vector_index:/app/vector_index \
  gal-rag-system
```

> 🔁 Use `-v $PWD/outputs:/app/outputs` if embedding in-container is also needed.

---

## 📁 Examples

* `examples/sample_input.txt` → describes input expectations
* `examples/sample_output.json` → shows normalized output format

---

## 🧪 Run Tests

```bash
pip install pytest
pytest tests/
```

---

## 🔐 Environment Variables

| Variable         | Description                                  |
| ---------------- | -------------------------------------------- |
| `OPENAI_API_KEY` | Your OpenAI API key (required for embedding) |

---

## 📃 License

MIT License