from typing import Union
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI, OpenAI
import os

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


def get_llm(
    model_name: str = None, llm_type: str = "openai"
) -> Union[ChatOpenAI, SentenceTransformer]:
    if llm_type != "openai":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return SentenceTransformer("all-MiniLM-L6-v2", device=device)

    return ChatOpenAI(model=model_name or chat_model_name, temperature=0)
    # return OpenAI(temperature=0)
