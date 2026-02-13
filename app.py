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
# GREETING & GRATITUDE HANDLER
# --------------------------------------------------
def detect_greeting_or_gratitude(text):
    """
    Detects if the message is a greeting, gratitude, or other common conversational phrases.
    Returns a tuple (is_greeting, response)
    """
    text_lower = text.lower().strip()
    
    # Gratitude responses
    gratitude_patterns = {
        'thank you': "You're welcome! Feel free to ask any more questions about admissions.",
        'thanks': "Happy to help! If you need any other information, just ask.",
        'thankyou': "You're welcome! Feel free to ask any more questions about admissions.",
        'thx': "You're welcome! Feel free to ask more admission-related questions.",
        'appreciate': "Glad I could help! Any other admission-related questions?",
        'thanks a lot': "You're welcome! Happy to assist with admission queries.",
        'thank you so much': "My pleasure! Feel free to ask if you need more information about admissions.",
    }
    
    # Greeting responses
    greeting_patterns = {
        'hello': "Hello! 👋 Welcome to the Army Institute of Technology Admission Assistant. How can I help you today with your admission queries?",
        'hi': "Hi there! 👋 I'm here to help answer your questions about AIT admissions. What would you like to know?",
        'hey': "Hey! 👋 Welcome! I can help you with questions about Army Institute of Technology admissions.",
        'hola': "¡Hola! 👋 I'm here to help with AIT admission questions. What would you like to know?",
        'good morning': "Good morning! 👋 I'm ready to help with your admission questions. What can I tell you?",
        'good afternoon': "Good afternoon! 👋 How can I assist you with AIT admissions today?",
        'good evening': "Good evening! 👋 I'm here to help with your admission queries.",
    }
    
    # Farewell responses
    farewell_patterns = {
        'bye': "Goodbye! 👋 Good luck with your admission process!",
        'goodbye': "Goodbye! 👋 All the best with your applications!",
        'see you': "See you! 👋 Feel free to reach out if you have more questions.",
        'see you later': "See you later! Good luck! 👋",
    }
    
    # Check gratitude patterns
    for pattern, response in gratitude_patterns.items():
        if pattern in text_lower:
            return True, response
    
    # Check farewell patterns
    for pattern, response in farewell_patterns.items():
        if pattern in text_lower:
            return True, response
    
    # Check greeting patterns
    for pattern, response in greeting_patterns.items():
        if text_lower == pattern or text_lower.startswith(pattern):
            return True, response
    
    return False, None


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

    # Check if the query is a greeting or gratitude
    is_greeting, greeting_response = detect_greeting_or_gratitude(query)

    with st.chat_message("assistant"):
        if is_greeting:
            # Respond to greeting/gratitude directly
            st.write(greeting_response)
            answer = greeting_response
        else:
            # Process as admission-related question
            with st.spinner("Searching official admission document..."):
                answer = admission_query_engine(query)
                st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
