import streamlit as st

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

load_dotenv()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Admission RAG Assistant", page_icon="🎓")
st.title("🎓 Army Institute of Technology Admission Assistant")


# --------------------------------------------------
# LOAD SYSTEM (ONLY ONCE)
# --------------------------------------------------
@st.cache_resource
def load_system():

    # 1️⃣ Load Admission PDF (no fitz)
    documents = SimpleDirectoryReader(
        input_files=["PROSPECTUS-2025.pdf"]
    ).load_data()

    # 2️⃣ Chunking
    splitter = SentenceSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    nodes = splitter.get_nodes_from_documents(documents)

    # 3️⃣ Embedding Model
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5"
    )

    # 4️⃣ Create Index
    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model
    )

    # 5️⃣ LLM (Groq - using llama-3.1-8b-instant)
    llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
        max_tokens=350
    )

    return index, llm


index, llm = load_system()


# --------------------------------------------------
# QUERY ENGINE
# --------------------------------------------------
def admission_query_engine(question):

    retriever = index.as_retriever(similarity_top_k=8)
    retrieved_nodes = retriever.retrieve(question)

    SIMILARITY_THRESHOLD = 0.25

    relevant_nodes = [
        node for node in retrieved_nodes
        if node.score is not None and node.score >= SIMILARITY_THRESHOLD
    ]

    if not relevant_nodes:
        return "I do not have sufficient information from official admission documents."

    top_nodes = relevant_nodes[:3]
    context = "\n\n".join([node.text for node in top_nodes])

    prompt = f"""
Answer the question strictly using the provided context.
Do not use outside knowledge.
Be concise and accurate.

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = llm.complete(prompt)
    except Exception as e:
        return f"LLM error: {e}"

    return str(response)


# --------------------------------------------------
# STREAMLIT CHAT UI
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
query = st.chat_input("Ask admission related question...")

if query:

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching official admission document..."):
            answer = admission_query_engine(query)
            st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
