import streamlit as st
import chromadb
from pypdf import PdfReader
from anthropic import Anthropic

st.set_page_config(page_title="Chat With Your Documents")
st.title("Chat With Your Documents")

api_key = st.sidebar.text_input("Anthropic API key", type="password")
chunk_size = st.sidebar.slider("Chunk size (characters)", 300, 1500, 800, 100)
top_k = st.sidebar.slider("Chunks to retrieve", 1, 6, 3)


def read_file(f):
    if f.name.lower().endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(f).pages)
    return f.read().decode("utf-8", errors="ignore")


def chunk_text(text, size, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


# Document -> Chunking -> Embeddings -> Vector DB
files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

if files and st.button("Process documents"):
    client = chromadb.Client()
    try:
        client.delete_collection("docs")
    except Exception:
        pass
    col = client.create_collection("docs")
    ids, texts, metas = [], [], []
    for f in files:
        for i, c in enumerate(chunk_text(read_file(f), chunk_size)):
            ids.append(f"{f.name}-{i}")
            texts.append(c)
            metas.append({"source": f.name})
    col.add(ids=ids, documents=texts, metadatas=metas)
    st.session_state["col"] = col
    st.success(f"Stored {len(texts)} chunks in the vector database.")

# Retrieval -> LLM -> Answer
question = st.text_input("Ask a question about your documents")

if question:
    if "col" not in st.session_state:
        st.warning("Upload and process a document first.")
    elif not api_key:
        st.warning("Add your API key in the sidebar.")
    else:
        res = st.session_state["col"].query(query_texts=[question], n_results=top_k)
        context = "\n\n---\n\n".join(res["documents"][0])
        prompt = (
            "Answer the question using only the context below. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        reply = Anthropic(api_key=api_key).messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        st.markdown("### Answer")
        st.write(reply.content[0].text)
        with st.expander("Retrieved chunks"):
            for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                st.caption(meta["source"])
                st.write(doc)
