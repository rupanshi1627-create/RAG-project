import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import tempfile

st.set_page_config(page_title="Document RAG Assistant", page_icon="📄")
st.title("📄 Document RAG Assistant")
st.write("Upload a document (PDF or DOCX) and ask questions about it. Answers are generated only from the document's content.")

# API key is read securely from Streamlit's secrets manager (never hardcoded)
groq_api_key = st.secrets["GROQ_API_KEY"]

uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])


@st.cache_resource(show_spinner="Processing document...")
def build_retriever(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = Docx2txtLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store.as_retriever(search_kwargs={"k": 3})


llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b")

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}"""
)

if uploaded_file is not None:
    file_type = "pdf" if uploaded_file.name.endswith(".pdf") else "docx"
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + file_type) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    retriever = build_retriever(tmp_path, file_type)
    st.success("Document processed! Ask your question below.")

    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Generating answer..."):
            docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            formatted_prompt = prompt.format(context=context, question=question)
            response = llm.invoke(formatted_prompt)

        st.subheader("Answer")
        st.write(response.content)

        with st.expander("View source chunks used (for transparency)"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Source {i + 1}:**")
                st.write(doc.page_content[:300])
