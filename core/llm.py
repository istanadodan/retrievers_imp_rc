import os
import pathlib
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings_model_name = os.getenv("EMBEDDINGS_MODEL")
chat_model_name = os.getenv("CHAT_MODEL")


def get_embeddings(model_name: str=None) -> "HuggingFaceEmbeddings":
    import torch

    model_name = (model_name or embeddings_model_name).lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encode_kwargs = {
        "normalize_embeddings": True,
        "device": device,
        # "show_progress_bar": True,
    }

    if 'all-MiniLM'  in model_name:

        # model_name = "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs=encode_kwargs,
            cache_folder=str(pathlib.Path("./models/sentence_transformers").resolve()),
            # multi_process=True,
        )
    elif 'jhgan' in model_name:

        # model_name = "jhgan/ko-sbert-nli"
        return HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs=encode_kwargs,
            cache_folder=str(pathlib.Path("./models/sentence_transformers").resolve()),
            # multi_process=True,
        )
    elif 'nomic' in model_name:
        from openai import OpenAI
        from langchain.embeddings.base import Embeddings
        class LMStudioEmbedding(Embeddings):
            def __init__(self, model:str, endpoint: str):
                self.model =model
                self.client = OpenAI(base_url=endpoint, api_key="lm-studio")
            
            def embed_documents(self, texts: list) -> list:
                embeddings = []
                for text in texts:
                     text = text.replace("\n", " ")
                     embedding = self.client.embeddings.create(input = text, model=self.model, encoding_format="base64").data[0].embedding                        
                     embeddings.append(embedding)
                print(f'{text=}, embeddings = {embedding}')
                return embeddings
            
            def embed_query(self, text: str) -> str:
                return self.embed_documents([text])
        
        base_url="http://localhost:1234/v1"

        return LMStudioEmbedding(model=model_name, endpoint=base_url)


def get_llm(model_name: str=None) -> "BaseChatModel":
    from langchain_community.llms.huggingface_hub import HuggingFaceHub
    from langchain_community.chat_models.huggingface import ChatHuggingFace

    # from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    model_name = (model_name or chat_model_name).lower()

    if "whisper" in model_name:
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

    elif "gpt" in model_name:
        return ChatOpenAI(model=model_name, temperature=0, max_tokens=500, verbose=True)
        # return OpenAI(temperature=0, max_tokens=500)

    elif "ollama" in model_name:
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(model="EEVE-Korean-10.8B:latest")
        
    elif "teddylee" in model_name:
        from langchain_openai.chat_models import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            streaming=True,
        )
