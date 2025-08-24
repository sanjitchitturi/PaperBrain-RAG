# Multimodal-RAG

*A Retrieval-Augmented Generation system that answers questions about a device using both its manual and annotated images.*  

## Overview  
This project demonstrates a **Multimodal RAG (Retrieval-Augmented Generation)** pipeline for device support and troubleshooting.  
Unlike standard RAG systems that retrieve only text, this demo:  

- Retrieves **textual evidence** from a device manual.  
- Retrieves **visual evidence** from device images.  
- Localizes the **exact button or region** in the photo relevant to the answer.  
- Provides **sourced snippets** from the manual alongside the highlighted image.  

This makes it easier for end users to **understand instructions visually and contextually**.  

---

## Features  
- **Manual-based Retrieval**: Finds relevant instructions from the device manual.  
- **Image-grounded Answers**: Highlights the button/region on the device photo.  
- **Evidence Attribution**: Each answer cites the manual snippet it came from.  
- **Multimodal QA**: Supports questions like *"How do I reset this device?"* or *"Which button is the power switch?"*.  

---

## Tech Stack  
- **LLM Backend**: GPT-based model for reasoning and generation.  
- **Vector Store**: FAISS / Weaviate / Pinecone (configurable).  
- **Document Processing**: PDF → text chunks for manual ingestion.  
- **Image Processing**: Grounding DINO / Segment Anything Model (SAM) for region detection.  
- **UI**: Streamlit or Gradio demo interface.  

---

## Project Structure  
```

multimodal-rag-demo/
│── data/                # Device manual (PDF/text) + reference images
│── notebooks/           # Exploration & prototyping
│── src/
│    ├── retriever.py    # Text + image retriever logic
│    ├── rag\_pipeline.py # Multimodal RAG pipeline
│    ├── ui.py           # Streamlit/Gradio app
│    ├── utils/          # Helper functions
│── requirements.txt     # Dependencies
│── README.md            # This file

````

---

## Installation  
1. Clone the repo:  
   ```bash
   git clone https://github.com/sanjitchitturi/multimodal-rag-demo.git
   cd multimodal-rag-demo
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install GPU-accelerated libraries for image models:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

---

## Usage

1. Add your **manual PDFs** to `data/manuals/`.
2. Add **device images** to `data/images/`.
3. Launch the demo UI:

   ```bash
   streamlit run src/ui.py
   ```
4. Ask questions like:

   * *"Where is the power button?"*
   * *"How do I change the batteries?"*

The app will:
Retrieve the manual snippet.
Highlight the relevant region in the photo.
Show both in a unified answer.

---

## Example Output

**Question:** *How do I reset the device?*

* **Manual Evidence:**

  > "To reset, press and hold the small recessed button labeled RESET on the back panel for 5 seconds."

* **Visual Evidence:**
  ![Highlighted Button Example](example_output/reset_highlight.png)

---
