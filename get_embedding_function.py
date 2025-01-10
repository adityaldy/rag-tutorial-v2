# from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings  # Updated import
from langchain_community.embeddings.bedrock import BedrockEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-multilingual-base",
        model_kwargs={"device": "cuda", "trust_remote_code": True},
    )
    # embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    # print(f"Using embedding model: {embeddings.model_name}")  # Add this line
    return embeddings
