# 🏥 End-to-End Medical Chatbot using Llama 2

A production-ready medical question-answering chatbot built with **Meta Llama 2**, **LangChain**, **Pinecone**, and **Flask** — using a Retrieval-Augmented Generation (RAG) pipeline grounded on a real medical knowledge base.

---

## 📋 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [RAG Pipeline Explained](#rag-pipeline-explained)
- [Files to Upload / Not Upload](#files-to-upload--not-upload)

---

## 📌 Overview

This chatbot answers medical questions using a **RAG (Retrieval-Augmented Generation)** approach:

1. A medical PDF book is chunked and embedded into a **Pinecone** vector store
2. When a user asks a question, the most relevant chunks are retrieved
3. **Llama 2 (7B quantized)** generates a grounded answer using only that retrieved context
4. The response is served via a **Flask** web app with a clean chat UI

This prevents hallucination — the model only answers from verified medical text, and says "I don't know" if the answer isn't in the source.

---

## 🏗️ System Architecture

```
User Question (Browser)
        │
        ▼
  Flask Web App (app.py)
        │
        ▼
  LangChain RetrievalQA Chain
        │
   ┌────┴────────────────────┐
   │                         │
   ▼                         ▼
Pinecone Vector DB      Llama 2 (7B Quantized)
(Semantic Search)       CTransformers (local)
   │
   ▼
Top-2 Relevant
Medical Text Chunks
        │
        └──────────────────► Prompt Template
                                    │
                                    ▼
                            Final Answer → User
```

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🧠 LLM | Meta Llama 2 7B Chat (quantized GGML, runs locally) |
| 📚 Knowledge Base | Medical book PDF — chunked into 500-token segments |
| 🔍 Retrieval | Pinecone vector DB with semantic similarity search |
| 🤗 Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| 🌐 Web Interface | Flask + Bootstrap chat UI with real-time AJAX responses |
| 🛡️ Hallucination Control | Prompt instructs model to say "I don't know" if answer not in context |
| ⚙️ Inference | CTransformers — CPU-friendly quantized model (q4_0) |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat&logo=chainlink&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-000000?style=flat)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Meta](https://img.shields.io/badge/Llama_2-0467DF?style=flat&logo=meta&logoColor=white)

| Component | Technology |
|---|---|
| LLM | Meta Llama 2 7B Chat (GGML q4_0 quantized) |
| Inference Engine | CTransformers 0.2.5 |
| RAG Framework | LangChain 0.0.225 |
| Vector Database | Pinecone |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Web Framework | Flask |
| Frontend | HTML5, Bootstrap 4, jQuery (AJAX) |
| PDF Loader | PyPDFLoader (LangChain) |

---

## 📁 Project Structure

```
End-to-end-Medical-Chatbot-using-Llama2/
│
├── src/
│   ├── __init__.py           # Package init
│   ├── helper.py             # PDF loader, text splitter, embeddings
│   └── prompt.py             # RAG prompt template for Llama 2
│
├── templates/
│   └── chat.html             # Chat UI (Bootstrap + jQuery AJAX)
│
├── static/
│   └── style.css             # Chat interface styling
│
├── data/
│   └── Medical_book.pdf      # Source knowledge base (medical reference)
│
├── model/
│   └── instruction.txt       # Instructions to download Llama 2 model
│   └── llama-2-7b-chat.ggmlv3.q4_0.bin   ← download separately (3.8 GB)
│
├── research/
│   └── trials.ipynb          # Experimentation notebook
│
├── app.py                    # Flask app — loads model, serves chat endpoint
├── store_index.py            # One-time script: embed PDF → Pinecone
├── template.py               # Project scaffolding script
├── setup.py                  # Package setup
├── requirements.txt          # Python dependencies
├── .env                      # Pinecone API keys (NOT committed to GitHub)
├── .gitignore                # Excludes .env, model binary, __pycache__
└── README.md
```

---

## ▶️ How to Run

### Step 1 — Clone the repository
```bash
git clone https://github.com/pawanahirwar/End-to-end-Medical-Chatbot-using-Llama2.git
cd End-to-end-Medical-Chatbot-using-Llama2
```

### Step 2 — Create and activate conda environment
```bash
conda create -n mchatbot python=3.8 -y
conda activate mchatbot
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Set up Pinecone credentials
Create a `.env` file in the root directory:
```ini
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_API_ENV = "your-pinecone-environment"
```
Get your free API key at [pinecone.io](https://www.pinecone.io/)

> **Note:** Create a Pinecone index named `medical-bot` with **384 dimensions** (matches `all-MiniLM-L6-v2` output)

### Step 5 — Download the Llama 2 model
Download the quantized model (~3.8 GB) and place it in the `model/` directory:
```
File: llama-2-7b-chat.ggmlv3.q4_0.bin
From: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
```

### Step 6 — Build the vector index (run once)
```bash
python store_index.py
```
This loads `Medical_book.pdf`, splits it into 500-token chunks, generates embeddings, and stores them in Pinecone.

### Step 7 — Launch the chatbot
```bash
python app.py
```

Open your browser at: **http://localhost:8080**

---

## 🔍 RAG Pipeline Explained

### Indexing (store_index.py — run once)
```
Medical_book.pdf
      │
      ▼ PyPDFLoader
Raw Text Pages
      │
      ▼ RecursiveCharacterTextSplitter
         chunk_size=500, chunk_overlap=20
Text Chunks (~500 tokens each)
      │
      ▼ HuggingFaceEmbeddings
         model: all-MiniLM-L6-v2 → 384-dim vectors
Dense Vectors
      │
      ▼ Pinecone.from_texts()
Stored in Pinecone Index: "medical-bot"
```

### Inference (app.py — every query)
```
User Question
      │
      ▼ HuggingFaceEmbeddings (same model)
Query Vector (384-dim)
      │
      ▼ Pinecone similarity search (top k=2)
2 Most Relevant Text Chunks
      │
      ▼ PromptTemplate
"Use the following context to answer..."
      │
      ▼ CTransformers → Llama 2 7B
         max_new_tokens=512, temperature=0.8
Final Answer
      │
      ▼ Flask /get endpoint → AJAX → Chat UI
```

### Prompt Template
```
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
```

---

## 📤 Files to Upload / Not Upload to GitHub

### ✅ Upload these files
| File | Reason |
|---|---|
| `app.py` | Main Flask application |
| `store_index.py` | Vector indexing script |
| `template.py` | Project scaffolding |
| `setup.py` | Package setup |
| `requirements.txt` | Dependencies |
| `src/__init__.py` | Package init |
| `src/helper.py` | Core helper functions |
| `src/prompt.py` | Prompt template |
| `templates/chat.html` | Chat UI |
| `static/style.css` | Styling |
| `data/Medical_book.pdf` | Knowledge base |
| `model/instruction.txt` | Model download instructions |
| `research/trials.ipynb` | Research notebook |
| `.gitignore` | Git ignore rules |
| `README.md` | This file |

### ❌ DO NOT upload these
| File | Reason |
|---|---|
| `.env` | Contains secret API keys — already in `.gitignore` |
| `model/llama-2-7b-chat.ggmlv3.q4_0.bin` | 3.8 GB binary — already in `.gitignore` |
| `__pycache__/` | Compiled bytecode — already in `.gitignore` |
| `*.egg-info/` | Build artifacts |

---

## ⚠️ Important Notes

- The Llama 2 model file is **3.8 GB** — it must be downloaded separately and is excluded from the repo via `.gitignore`
- The `.env` file with your Pinecone keys must **never** be committed
- `store_index.py` only needs to be run **once** to populate Pinecone
- The model runs on **CPU** via CTransformers — response time depends on your hardware (GPU recommended for faster inference)

---

## 👤 Author

**Pawan Singh Ahirwar**
- GitHub: [@pawanahirwar](https://github.com/pawanahirwar)
- Affiliation: IRCC, IIT Bombay
