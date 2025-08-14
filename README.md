### 🌍 Live Demo

This project is fully **Dockerized** and deployed on an **AWS EC2 instance**:

🔗 **Try it here**: [http://65.2.113.254:8501/](http://65.2.113.254:8501/)

* The chatbot is pre-loaded with sample legal documents from the `samples/` folder, which acts as the **knowledge base**.
* Users can ask questions based on the contents of these files — judgments, filings, rulings, etc.
* Responses include **contextual answers** retrieved via FAISS from chunked and embedded documents.

> 🚀 No signup needed — just ask a legal question to get started.

Sample Conversation - 

![image](https://github.com/user-attachments/assets/23ca8095-3ef1-41bc-b2fa-3ee46b3006ea)

---

## 🧠 Superlaw – Legal Document RAG Chatbot

Superlaw is a Retrieval-Augmented Generation (RAG) chatbot built to help users query legal documents using natural language. Powered by FAISS, OpenAI embeddings, and Streamlit, it allows instant Q\&A from uploaded legal files.

---

### 🚀 Features

* 📄 Upload and parse legal documents (`PDF`, `DOCX`, `TXT`, etc.)
* 🔍 Chunk content and create a FAISS vector index
* 🧠 Generate embeddings using OpenAI's `text-embedding-3-small`
* 💬 Ask questions and get contextual answers with source citations
* 🖥️ Web interface powered by Streamlit
* 📦 Containerized via Docker for easy deployment (e.g., EC2)

---

### 📁 Project Structure

```
superlaw/
│
├── samples/               # Input documents (PDFs, DOCX, etc.)
├── vector_index/          # Generated FAISS index and metadata
├── outputs/               # Chunked outputs (optional)
├── src/
│   ├── app.py             # Streamlit frontend
│   ├── parser.py          # File parsing via unstructured
│   ├── chunker.py         # Chunking logic
│   ├── embed_and_store.py # Embedding + FAISS index creation
│   ├── rag_chat.py        # RAG logic + LLM response
│   └── config.yaml        # Config (optional)
├── requirements.txt
└── Dockerfile
```

---

### 🧰 Requirements

* Python 3.11
* OpenAI API Key (set in `.env`)
* Docker (for deployment)

---

### 🛠️ Local Setup

```bash
git clone https://github.com/vjvashisth/superlaw.git
cd superlaw
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your API key in `.env`:

```
OPENAI_API_KEY=sk-...
```

Run locally:

```bash
streamlit run src/app.py
```

---

### 🐳 Docker Deployment

Build image:

```bash
docker build -t superlaw-chatbot .
```

Run container:

```bash
docker run -d -p 8501:8501 --env-file .env superlaw-chatbot
```

> ⚠️ Ensure `vector_index/` is included in the image or mounted via volume.

---

### 🚧 Enhancements Roadmap

Planned upgrades to improve Superlaw’s capabilities and user experience:

1. **📚 Incremental Vector DB Updates**
   Enable automatic or scheduled ingestion of new judgments to keep the knowledge base current.

2. **🔗 API-Based Source Integration**
   Pull legal documents via APIs (e.g., Indian Kanoon, Manupatra, or Court websites) for seamless ingestion and indexing.

3. **🧾 Citation Highlighting**
   Visually highlight which parts of the retrieved context were used to generate a response.

4. **🔒 User Authentication & Access Control**
   Set up OAuth or JWT-based authentication with role-based access for legal teams, firms, and premium users.

5. **💎 Premium User Features**
   Allow premium users to download referenced judgment copies, receive long-form summaries, or access GPT-4-turbo.

6. **☁️ Cloud-Native Deployment**
   Deploy to AWS/GCP/Azure using scalable services (e.g., ECS, EKS, Lambda + API Gateway) and CDN-backed UI.

7. **🧠 LLM Experimentation Framework**
   Switch easily between GPT-4, Claude, Mistral, Llama, or local open-source models using LangChain or LlamaIndex wrappers.

8. **🌐 Web, Mobile & App Extensions**
   Build public-facing UI: responsive web portal, Android/iOS apps, and browser extensions to make Superlaw universally accessible.

9. **📊 Dashboard for Usage Analytics**
   Track queries, document uploads, and usage metrics for admin and legal ops teams.

10. **🗃️ Case-Type Based Indexing**
    Organize vector DBs by case type (e.g., criminal, civil, tax, IP) for targeted legal research.

11. **🎯 Intent-aware Answer Generation**
    Add semantic query classification (e.g., facts, ruling, precedent, procedure) to tailor answers precisely.

12. **🛡️ Legal Risk & Bias Detection**
    Include risk disclaimers, fairness evaluations, and hallucination checks for compliance-grade reliability.

---

### 📜 License

MIT © 2025 Vijayendra Vashisth
