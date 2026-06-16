import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import tempfile

api_key = "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"

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=api_key
)

st.title("PDF Chatbot")
st.write("Upload a PDF and ask questions about it")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        temp_path = f.name

    with st.spinner("Reading PDF..."):
        loader = PyPDFLoader(temp_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(pages)
        full_text = "\n".join([chunk.page_content for chunk in chunks])

    st.success("PDF loaded! Ask your questions below.")

    question = st.text_input("Ask a question about your PDF:")

    if question:
        with st.spinner("Thinking..."):
            response = llm.invoke(
                f"Based on this document:\n\n{full_text[:6000]}\n\nAnswer this question: {question}"
            )
            st.write("**Answer:**", response.content)