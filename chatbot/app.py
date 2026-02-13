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
# Load ENV Variables
# -------------------------------
load_dotenv()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Laptop Support AI Assistant")
st.title("💻 Intelligent AI Assistant for Tech Support")

PDF_FOLDER = "D:\\Learnathon\\rawdocs"

# -------------------------------
# Load Knowledge Base
# -------------------------------
@st.cache_resource
def load_knowledge_base():

    documents = []

    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, file)

            loader = PyPDFLoader(pdf_path)
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

    # ✅ Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    prompt = PromptTemplate.from_template("""
    You are an enterprise laptop support assistant.

    Use the following retrieved context to answer the question accurately.

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
        return_source_documents=True,
    )

    return qa_chain


qa_chain = load_knowledge_base()

# -------------------------------
# Chat Memory
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# User Input
# -------------------------------
user_question = st.text_input("Describe the laptop issue:")

if user_question:

    result = qa_chain({
        "question": user_question,
        "chat_history": st.session_state.chat_history
    })

    st.session_state.chat_history.append(
        (user_question, result["answer"])
    )

# -------------------------------
# Display Chat
# -------------------------------
for question, answer in st.session_state.chat_history:
    st.write(f"**👨‍💻 You:** {question}")
    st.write(f"**🤖 Assistant:** {answer}")
