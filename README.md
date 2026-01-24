# 📄 Chat with Multiple PDFs (Local RAG App)

A local **Retrieval-Augmented Generation (RAG)** application that allows users to upload multiple PDF documents and ask questions grounded **only in the provided documents**.

The system uses **semantic search (FAISS)** combined with a **local LLM (Ollama – Mistral)** to generate accurate, context-aware answers without relying on paid APIs.

---

## 🚀 Features

- 📚 Upload and chat with **multiple PDFs**
- 🔍 Semantic search using **FAISS vector store**
- 🧠 Multi-turn conversation with **explicit memory**
- 🤖 Runs on **local LLM (Ollama – Mistral)**  
- 🖥️ Simple **Streamlit UI**
- 🔒 Answers restricted to uploaded documents (hallucination-controlled)

---

## 🧠 How It Works

1. **PDF Ingestion**
   - Extracts raw text from uploaded PDF files

2. **Chunking**
   - Splits text into overlapping semantic chunks

3. **Embeddings**
   - Converts text chunks into vector embeddings using HuggingFace models

4. **Vector Store**
   - Stores embeddings in a FAISS index for fast similarity search

5. **Retrieval**
   - Retrieves the most relevant chunks for a user query

6. **Generation**
   - Sends retrieved context + chat history to a local LLM (Mistral via Ollama)

7. **Conversation Memory**
   - Maintains multi-turn chat context explicitly (no hidden state)

---

## 🧩 Tech Stack

- **Python**
- **Streamlit** – UI
- **LangChain (LCEL, latest)** – orchestration
- **FAISS** – vector similarity search
- **HuggingFace Embeddings**
- **Ollama (Mistral)** – local LLM
- **PyPDF** – PDF parsing

---

## 📂 Project Structure

```text
Multiple_PDF_Chat/
│
├── app.py                # Streamlit application
├── HTML_templates.py     # HTML/CSS templates for chat UI
├── README.md
├── requirements.txt
│
└── .venv/                # Virtual environment (not committed)
```

---

## ⚙️ Setup Instructions
1. **Clone the Repository**
```bash
git clone <repository-url>
cd Multiple_PDF_Chat
```

2. **Create and Activate Virtual Environment**
```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Install and Run Ollama**

Download Ollama from:
https://ollama.com/download

Pull the Mistral model:

```bash
ollama pull mistral
```

Start Ollama:

```bash
ollama run mistral
```

5. **Run the Application**
```bash
streamlit run app.py
```

## Open the app in your browser at:

http://localhost:8501

## 🧪 Usage

```text
Upload one or more PDF documents

Click Process

Ask questions related to the uploaded PDFs

Ask follow-up questions — the system remembers context

If the answer is not found in the documents, the assistant responds:

I don't know
```

## 🧠 Design Principles

> Explicit memory management

> Retriever-first grounding

> UI separated from AI logic

> Model-agnostic architecture

> Local-first execution

## 🔮 Future Improvements

<!-- Persist FAISS index to disk

Display source document chunks with answers

Add metadata-based filtering

Add evaluation and logging

Convert backend to FastAPI -->

## 📌 Disclaimer

### This project is for learning and demonstration purposes and is not intended for production use without additional hardening.

## 👤 Author

### Gaurav
Built as part of an Applied AI / RAG learning journey.