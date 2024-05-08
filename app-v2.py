from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.vectorstores.faiss import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from helper import list_llm_models, get_embeddings, get_llm
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.title("News Research ChatBot")
main_progress_bar = st.empty()
embedding_type = None
model_id = None

# Conversation chain
def get_context_retriever_chain(retriever):
    llm = get_llm(model_id)

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to this conversation")
    ])

    main_progress_bar.text(":green[██████] 30%")
    history_document = create_history_aware_retriever(llm, retriever, prompt)

    return history_document

# RAG Chain
def get_conversational_rag_chain(history_document):
    llm = get_llm(model_id)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    main_progress_bar.text(":green[█████████] 45%")
    stuff_documents_chains = create_stuff_documents_chain(llm, prompt)

    main_progress_bar.text(":green[████████████] 60%")
    retrieval_chain = create_retrieval_chain(history_document, stuff_documents_chains)

    return retrieval_chain

# Response
def get_response(user_query):
    history_document = get_context_retriever_chain(st.session_state.retriever)
    retrieval_chain = get_conversational_rag_chain(history_document)

    main_progress_bar.text(":green[████████████████] 78%")
    response = retrieval_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    return response["answer"]

with st.sidebar:
    st.title("News Article URLs")
    if 'url_count' not in st.session_state:
        st.session_state.url_count = 3
    
    button_plus_clicked, button_minus_clicked = st.columns([0.5,2.0])

    with button_plus_clicked:
        button_plus_clicked = st.button("➕")
    
    with button_minus_clicked:
        button_minus_clicked = st.button("➖")

    if button_plus_clicked:
        st.session_state.url_count += 1

    if button_minus_clicked:
        st.session_state.url_count -= 1
        if st.session_state.url_count == 0:
            st.session_state.url_count = 1

    urls = []
    for i in range(st.session_state.url_count):
        url = st.text_input(f"URL {i+1}")
        urls.append(url)

    embedding_type = st.radio("Select embedding model", ("Paid OpenAI", "Free Google", "Free Huggingface"))
    model_id = st.selectbox("Select LLM", list_llm_models())
    process_url_clicked = st.button("Process URL")
    sidebar_progress = st.empty()

    if embedding_type and process_url_clicked:
        # Load Data
        loader = WebBaseLoader(web_path=urls)
        sidebar_progress.text("Data Loading Started...")
        data = loader.load()
        if not data:
            sidebar_progress.text("Error: No data loaded from URLs. Please check URLs.")
            st.stop()
        # Split Data
        text_spliter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " "], chunk_size=1000)
        sidebar_progress.text("Text Splitting Started...")
        docs = text_spliter.split_documents(data)
        # Create Embeddings and Save it in local
        embeddings = get_embeddings(embedding_type)
        sidebar_progress.text("Embedding Vector Started Building...")
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        vectorstore_openai.save_local(folder_path=embedding_type)
        sidebar_progress.empty()

if os.path.exists(embedding_type):

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?")
        ]
    if 'retriever' not in st.session_state:
        embeddings = get_embeddings(embedding_type)
        vectorstore = FAISS.load_local(folder_path=embedding_type, embeddings=embeddings, allow_dangerous_deserialization=True)
        st.session_state.retriever = vectorstore.as_retriever()

    query = st.chat_input("Type your query here...")

    if query is not None and query != "":
        response = get_response(query)
        st.session_state.chat_history.append(HumanMessage(content=query))
        st.session_state.chat_history.append(AIMessage(content=response))

    main_progress_bar.text(":green[████████████████████] 99%")
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
    main_progress_bar.empty()