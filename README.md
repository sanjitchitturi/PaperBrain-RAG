## PaperBrain-RAG

Perfect, here’s a **ready-to-use README.md** you can copy-paste straight into your GitHub repo. I’ll make it professional, beginner-friendly, and recruiter-attractive.

---

# Smart Document Q\&A (RAG Demo)

A simple **Retrieval-Augmented Generation (RAG)** system that lets you upload a PDF manual (or any document) and **ask natural language questions** about it.

This project demonstrates how to build a lightweight RAG pipeline **without external APIs** — everything runs locally on CPU.

---

## Features

* **PDF Ingestion** → Extract text from any PDF.
* **Chunking** → Split text into manageable, sentence-based chunks.
* **Embeddings + FAISS** → Convert chunks into vectors and store them for fast search.
* **Question Answering** → Retrieve the most relevant snippets for a user’s query.
* **Streamlit UI** → Clean and interactive web interface.
* **Offline** → No API calls, no GPU needed, runs on a simple laptop.
  
---

## Project Structure

```
smart-doc-rag/
│── data/manuals/     
│── src/
│    ├── retriever.py   
│    ├── rag_pipeline.py 
│    ├── ui.py         
│── requirements.txt   
│── README.md           
```

---

## Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/sanjitchitturi/smart-doc-rag.git
cd smart-doc-rag
pip install -r requirements.txt
```

Dependencies:

* [streamlit](https://streamlit.io/) → simple UI
* [pypdf](https://pypi.org/project/pypdf/) → PDF text extraction
* [faiss-cpu](https://github.com/facebookresearch/faiss) → vector search
* [sentence-transformers](https://www.sbert.net/) → text embeddings

---

## Usage

1. Place your **PDF manual** inside `data/manuals/`.
   (or upload via the Streamlit UI).

2. Run the app:

   ```bash
   streamlit run src/ui.py
   ```

3. Open the Streamlit link in your browser.

4. Upload a PDF, then type a question like:

   * *“How do I reset this device?”*
   * *“Where is the power button?”*
   * *“How do I change the batteries?”*

5. The app will display the **most relevant snippets** from the manual.

---

## Example Output

**Question:** *How do I reset the device?*

**Answer:**

```
1. To reset the device, press and hold the recessed RESET button on the back panel for 5 seconds.  

2. If the device is unresponsive, disconnect the power and try again.  
```

---

## How It Works

1. **PDF Processing** → Extract text and split into sentence-based chunks.
2. **Embeddings** → Use a local transformer (`all-MiniLM-L6-v2`) to convert chunks into vectors.
3. **FAISS Indexing** → Store embeddings in FAISS for fast nearest-neighbor search.
4. **Retrieval** → On a query, embed the question and fetch the most similar text chunks.
5. **Answer** → Display the retrieved snippets in the UI.

---

## Future Improvements

* Add **page numbers** for each snippet.
* Add **chat history** for multi-turn Q\&A.
* Extend to **Multimodal RAG** (highlight regions in images).
* Deploy to **Streamlit Cloud** for easy sharing.

---
