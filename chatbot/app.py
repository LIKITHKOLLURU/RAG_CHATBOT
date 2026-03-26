import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------
# Load ENV
# -------------------------------
load_dotenv()

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Laptop Support AI",
    page_icon="💻",
    layout="wide"
)

st.title("💻 Intelligent AI Laptop Support Assistant")
st.caption("AI-powered enterprise laptop troubleshooting system")

PDF_FOLDER = "D:\\Learnathon\\rawdocs"

# -------------------------------
# Knowledge Base
# -------------------------------
@st.cache_resource
def load_knowledge_base():

    documents = []

    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    prompt = PromptTemplate.from_template("""
You are an enterprise laptop support assistant.

Answer ONLY using the provided context.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
""")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        combine_docs_chain_kwargs={"prompt": prompt},
    )

    return qa_chain


qa_chain = load_knowledge_base()

# -------------------------------
# Chat Memory
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.info(
        "This assistant diagnoses laptop issues using enterprise PDF knowledge."
    )

# -------------------------------
# Display Previous Messages
# -------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------
# Chat Input
# -------------------------------
user_prompt = st.chat_input("Describe your laptop issue...")

if user_prompt:

    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Diagnosing laptop issue... 🔍"):

            chat_history = [
                (m["content"], "")
                for m in st.session_state.messages
                if m["role"] == "user"
            ]

            result = qa_chain({
                "question": user_prompt,
                "chat_history": chat_history
            })

            response = result["answer"]
            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
