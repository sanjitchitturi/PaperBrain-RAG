import os
import streamlit as st
from rag_pipeline import RagPipeline

APP_TITLE = "PaperBrain RAG"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
MANUALS_DIR = os.path.join(DATA_DIR, "manuals")

def save_upload(file) -> str:
    os.makedirs(MANUALS_DIR, exist_ok=True)
    path = os.path.join(MANUALS_DIR, file.name)
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    return path

def show_snippet(r, i):
    with st.container():
        st.markdown(f"**#{i} · Score {r['score']:.3f} · Page {r['page']}**")
        st.write(r["text"])

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Upload a PDF and ask questions. Runs fully offline on CPU.")

    with st.sidebar:
        st.header("1. PDF selection")
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
        existing = [f for f in os.listdir(MANUALS_DIR)] if os.path.exists(MANUALS_DIR) else []
        picked = st.selectbox("Or pick existing", ["(none)"] + existing)

        st.header("2. Settings")
        top_k = st.slider("Snippets to return", 1, 10, 5)
        rebuild = st.checkbox("Rebuild index", False)

    pipe = RagPipeline(DATA_DIR)

    pdf_path = None
    if uploaded:
        try:
            pdf_path = save_upload(uploaded)
            st.success(f"Uploaded: {os.path.basename(pdf_path)}")
        except Exception as e:
            st.error(f"Failed to save upload: {e}")
            pdf_path = None
    elif picked != "(none)":
        pdf_path = os.path.join(MANUALS_DIR, picked)

    st.header("Ask a question")
    q = st.text_input("Example: 'How do I reset this device?'", value="")
    search_btn = st.button("Search")

    if search_btn:
        if not pdf_path:
            st.warning("Please upload or select a PDF first.")
        elif not q.strip():
            st.warning("Please enter a question.")
        else:
            try:
                if rebuild:
                    with st.spinner("Rebuilding index..."):
                        pipe.ensure_index(pdf_path, rebuild=True)
                with st.spinner("Searching..."):
                    results = pipe.ask(pdf_path, q, k=top_k)

                if results:
                    st.subheader("Matches")
                    for i, r in enumerate(results, 1):
                        show_snippet(r, i)
                    st.subheader("Concatenated Answer")
                    st.write("\n\n".join(r["text"] for r in results))
                else:
                    st.warning("No matches found.")
            except Exception as e:
                st.error(f"Error during search: {e}")

    with st.expander("About"):
        st.markdown(
            """
        **How it works**
        1. Extract text from PDF
        2. Split into sentence chunks
        3. Embed chunks with `all-MiniLM-L6-v2`
        4. Store in FAISS index (cosine sim)
        5. Query → embed → retrieve top matches

        Runs 100% locally, CPU only.
        """
        )

if __name__ == "__main__":
    main()
