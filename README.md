# PaperBrain RAG

PaperBrain RAG is a lightweight **Retrieval-Augmented Generation (RAG)** system that lets you upload a PDF (manuals, research papers, guides, etc.) and ask natural language questions about it.  

Everything runs **locally on CPU** — no API keys, no GPU, no external services.  

---

## Features

- **PDF Ingestion** → Extracts text from any PDF  
- **Sentence-based Chunking** → Splits text into readable pieces  
- **FAISS Index** → Efficient vector similarity search  
- **Local Embeddings** → Uses `all-MiniLM-L6-v2` from Sentence Transformers  
- **Streamlit UI** → Clean, interactive interface  
- **Offline** → Runs on a regular laptop, no API calls  

---

## Project Structure

```

PaperBrain-RAG/
├── data/
│   ├── manuals/      
│   └── indexes/      
├── src/
│   ├── retriever.py
│   ├── rag\_pipeline.py
│   └── ui.py  
├── requirements.txt
└── README.md

````

---

## Installation

Clone and install dependencies:

```bash
git clone https://github.com/sanjitchitturi/PaperBrain-RAG.git
cd PaperBrain-RAG
pip install -r requirements.txt
````
---

## Usage

1. Put your PDF inside `data/manuals/` (or upload via the UI).
2. Start the app:

```bash
cd src
streamlit run ui.py
```

3. Open the Streamlit link in your browser.
4. Ask questions like:

   * *"How do I reset this device?"*
   * *"Where is the power button?"*
   * *"How do I change the batteries?"*

---

## How It Works

1. **PDF Processing** → Extract text per page, clean + split into sentence-based chunks
2. **Embeddings** → Chunks → vectors via `all-MiniLM-L6-v2` (runs on CPU)
3. **FAISS Indexing** → Vectors stored in FAISS for fast similarity search
4. **Retrieval** → Question → embed → nearest neighbor search
5. **Answer** → Top snippets are returned & concatenated for context

---
