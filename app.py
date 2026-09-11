import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import tempfile

st.set_page_config(page_title="Document RAG Assistant 🌼", page_icon="🌼")
st.title("📄 Document RAG Assistant")
st.write("Upload a document (PDF or DOCX) and ask questions — including follow-ups!")

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

# Prompt ab conversation history bhi include karta hai, taaki
# follow-up questions ("uska", "wo", "iska") samajh aa sakein
prompt = ChatPromptTemplate.from_template(
    """You are answering questions about a document. Use the retrieved context
and the conversation history below to answer the latest question. If the new
question refers to something from earlier (like "it", "he", "that"), use the
history to understand what it means.

Conversation history so far:
{history}

Retrieved context from the document:
{context}

New question: {question}

Answer:"""
)

# session_state Streamlit ka special storage hai jo reruns ke beech data
# preserve karta hai. Bina isके, har naye sawal pe purana chat history
# gayab ho jaata (kyunki Streamlit poora script dobara chalata hai har action pe)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # (question, answer) tuples ki list

if uploaded_file is not None:
    file_type = "pdf" if uploaded_file.name.lower().endswith(".pdf") else "docx"
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + file_type) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    retriever = build_retriever(tmp_path, file_type)
    st.success("Document processed! Ask your question below.")

    # Ab tak ka poora conversation dikhao (chat bubbles ki tarah)
    for past_question, past_answer in st.session_state.chat_history:
        st.chat_message("user").write(past_question)
        st.chat_message("assistant").write(past_answer)

    # Naya chat-style input box (neeche fixed rehta hai, WhatsApp jaisa feel)
    question = st.chat_input("Ask a question (follow-ups work too!)")

    if question:
        st.chat_message("user").write(question)

        with st.spinner("Generating answer..."):
            docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])

            # Pichhle saare Q&A ko ek readable text mein convert kar rahe hain
            if st.session_state.chat_history:
                history_text = "\n".join(
                    [f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history]
                )
            else:
                history_text = "No previous questions yet."

            formatted_prompt = prompt.format(
                history=history_text, context=context, question=question
            )
            response = llm.invoke(formatted_prompt)

        st.chat_message("assistant").write(response.content)

        # Ye naya exchange bhi history mein save karo, taaki AGLA
        # follow-up question isko bhi reference kar sake
        st.session_state.chat_history.append((question, response.content))
