from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_core.messages import AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.vectorstores.faiss import FAISS
from helper import list_llm_models, get_embeddings, get_llm
from htmlTemplate import user_template, bot_template, css
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
st.write(css, unsafe_allow_html=True)
st.subheader(":orange[News research chat bot]🤖")
main_progress_bar = st.empty()
embedding_type = None
model_id = None

with st.sidebar:

    st.subheader(":blue[News article URLs]")
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
        if url:
            urls.append(url)

    embedding_type = st.radio("Select embedding model", ("Paid OpenAI", "Free Google", "Free Huggingface"))
    model_id = st.selectbox("Select LLM", list_llm_models())
    process_url_clicked = st.button("Process URL")
    sidebar_progress = st.empty()

    if embedding_type and process_url_clicked:
        
        if len(urls)==0:
            sidebar_progress.text(":red[Resource not supplied or not accessible...]")
            st.stop()
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

def create_conversation_chain():

    llm = get_llm(model_id)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.retriever,
        memory=st.session_state.memory
    )

    return conversation_chain

def handle_userinput():
    response = st.session_state.conversation
    for element in response['chat_history']:
        st.session_state.chat_history.append(element)

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

if os.path.exists(embedding_type):

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]
    if 'retriever' not in st.session_state:
        embeddings = get_embeddings(embedding_type)
        vectorstore = FAISS.load_local(folder_path=embedding_type, embeddings=embeddings, allow_dangerous_deserialization=True)
        st.session_state.retriever = vectorstore.as_retriever()

    query = st.chat_input("Type your query here...")

    if query and query != "":
        chain = create_conversation_chain()
        st.session_state.conversation = chain({"question":query})
        print(chain.memory.buffer)
        handle_userinput()
    elif st.session_state.conversation is None:
        st.write(bot_template.replace("{{MSG}}", st.session_state.chat_history[0].content), unsafe_allow_html=True)
    elif len(st.session_state.chat_history)>1:
        handle_userinput()
    main_progress_bar.empty()