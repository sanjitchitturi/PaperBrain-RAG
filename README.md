# PaperBrain RAG

**PaperBrain RAG** is a lightweight Retrieval Augmented Generation system that lets you upload a PDF and ask natural language questions about it.  

Everything runs **locally on CPU** — no API keys, no GPU, no external services. Just drop in a PDF and start asking questions.

---

## Features
- **PDF Ingestion** → Extracts text from any PDF. 
- **Sentence-based Chunking** → Splits text into readable pieces.  
- **FAISS Index** → Fast similarity search over embeddings.  
- **Local Embeddings** → Uses `all-MiniLM-L6-v2` from Sentence Transformers.  
- **Streamlit UI** → Clean, interactive interface.  
- **Offline** → Runs on a regular laptop, no API calls required.  

---

## Project Structure
```

paperbrain-rag/
├── data/
│   ├── indexes
│   └── manuals    
├── src/
│   ├── rag_pipeline.py 
│   ├── retriever.py 
│   └── ui.py             
├── requirements.txt
└── README.md

````

---

## Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/sanjitchitturi/paperbrain-rag.git
cd paperbrain-rag
pip install -r requirements.txt
````

---

## Usage

1. Place your PDF in `data/manuals/`.
2. Start the app:

```bash
cd src
streamlit run ui.py
```

3. Open the Streamlit link in your browser.

4. Ask natural language questions such as:

   * *How do I reset this device?*
   * *Where is the power button?*
   * *How do I change the batteries?*

---

## How It Works

1. **PDF Processing** → Extract text per page and split into sentence-based chunks.
2. **Embeddings** → Convert chunks into vectors with `all-MiniLM-L6-v2`.
3. **FAISS Indexing** → Store vectors in FAISS for fast similarity search.
4. **Retrieval** → Question → embed → nearest neighbor search.
5. **Answer** → Top snippets are returned and concatenated for context.

---

## Notes

* Runs fully offline, on CPU only.
* Works with manuals, guides, research papers, or any readable PDF.
* Indexes are cached in `data/indexes/` for faster repeated queries.

---
