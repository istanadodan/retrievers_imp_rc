from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

embeddings_model_name = os.getenv("embeddings_model_name")
chat_model_name = os.getenv("chat_model_name")


def get_embeddings(model_name: str = None) -> HuggingFaceEmbeddings:
    import pathlib

    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(
        model_name=model_name or embeddings_model_name,
        encode_kwargs=encode_kwargs,
        cache_folder=str(pathlib.Path("./models/sentence_transformers").resolve()),
    )


def get_llm(model_name: str = None) -> ChatOpenAI:
    return ChatOpenAI(model=model_name or chat_model_name, temperature=0)
