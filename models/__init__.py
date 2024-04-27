import os
import pathlib
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings_model_name = os.getenv("embeddings_model_name")
chat_model_name = os.getenv("chat_model_name")


def get_embeddings(model_name: str = "jhgan/ko-sbert-nli") -> "HuggingFaceEmbeddings":
    import torch

    model_name = model_name or embeddings_model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encode_kwargs = {
        "normalize_embeddings": True,
        "device": device,
        # "show_progress_bar": True,
    }

    if model_name.lower().find("all-MiniLM") >= 0:

        # model_name = "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs=encode_kwargs,
            cache_folder=str(pathlib.Path("./models/sentence_transformers").resolve()),
            # multi_process=True,
        )
    elif model_name.lower().find("jhgan") >= 0:

        # model_name = "jhgan/ko-sbert-nli"
        return HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs=encode_kwargs,
            cache_folder=str(pathlib.Path("./models/sentence_transformers").resolve()),
            # multi_process=True,
        )


def get_llm(model_name: str = None) -> "BaseChatModel":
    from langchain_community.llms.huggingface_hub import HuggingFaceHub
    from langchain_community.chat_models.huggingface import ChatHuggingFace

    # from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI, OpenAI

    model_name = model_name or chat_model_name

    if model_name.lower().find("whisper") >= 0:
        # python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('YOUR_TOKEN_HERE')"
        _llm = HuggingFaceHub(
            repo_id="openai/whisper-large-v3",
            task="text-generation",
            model_kwargs={
                "max_new_tokens": 128,
                "top_k": 10,
                "temperature": 0.1,
                "repetition_penalty": 1.03,
            },
        )
        return ChatHuggingFace(
            llm=_llm,
            verbose=True,
        )

    elif model_name.find("gpt") >= 0:
        return ChatOpenAI(model=model_name, temperature=0, max_tokens=500, verbose=True)
        # return OpenAI(temperature=0, max_tokens=500)
