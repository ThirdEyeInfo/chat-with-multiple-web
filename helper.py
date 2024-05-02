from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.llms.ollama import Ollama
 
def list_llm_models() :

    return ["Google Gemini Pro", "Meta LLAMA 2", "Meta LLAMA 3", "OpenAI GPT 3.5 Turbo", "OpenAI GPT 4 Turbo", "Huggingface Hub"]

def get_embeddings(embedding_type):

    if embedding_type == "Paid OpenAI":
        return OpenAIEmbeddings(model="text-embedding-3-large")
    elif embedding_type == "Free Google":
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    elif embedding_type == "Free Huggingface":
        return HuggingFaceHubEmbeddings(model="sentence-transformers/all-mpnet-base-v2")

def get_llm(model_id):

    if model_id == "OpenAI GPT 3.5 Turbo":
        return ChatOpenAI(model="gpt-3.5-turbo")
    elif model_id == "OpenAI GPT 4 Turbo":
        return ChatOpenAI(model="gpt-4-turbo")
    elif model_id == "Google Gemini Pro":
        return ChatGoogleGenerativeAI(model="gemini-pro")
    elif model_id == "Meta LLAMA 2":
        return Ollama(model="llama2")
    elif model_id == "Meta LLAMA 3":
        return Ollama(model="llama3")
    elif model_id == "Huggingface Hub":
        return HuggingFaceHub(repo_id="google/flan-t5-large")