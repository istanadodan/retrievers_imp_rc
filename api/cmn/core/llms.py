from typing import List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from api.cmn.settings import settings


def get_llm(type: str) -> BaseChatModel:
    if type == "local":
        return ChatOpenAI(
            base_url=settings.LOCAL_BASE_URL,
            api_key=settings.LOCAL_API_KEY,
            model=settings.LOCAL_LLM_MODEL,
            streaming=True,
            verbose=True,
            temperature=0.7,
            max_tokens=512,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0,
            # stop=["\n\n"],
            callbacks=[StreamingStdOutCallbackHandler()],
        )
    else:
        raise NotImplementedError(f"Unknown model type: {type}")


def get_embedding(type: str) -> BaseChatModel:
    if type == "local":
        return ChatOpenAI(
            base_url=settings.LOCAL_BASE_URL,
            api_key=settings.LOCAL_API_KEY,
            model=settings.LOCAL_EMBEDDING_MODEL,
            streaming=True,
            verbose=True,
        )
    else:
        raise NotImplementedError(f"Unknown model type: {type}")


def get_embedding(type: str):
    from langchain_core.documents import Document

    if type == "local":
        class LMStudioEmbedding:
            def __init__(self, model: str, endpoint: str, api_key: str):
                from openai import OpenAI

                self.model = model
                self.client = OpenAI(base_url=endpoint, api_key=api_key)

            def embed_documents(self, documents:list[Document]) -> list:
                embeddings = []
                for document in documents:            
                    text = document.page_content.replace("\n", " ")
                    embedding = (
                        self.client.embeddings.create(input=[text], model=self.model)
                        .data[0]
                        .embedding
                    )
                    embeddings.append(embedding)
                return embeddings

            def embed_query(self, text: str) -> List[float]:
                return self.embed_documents([Document(page_content=text)])

        return LMStudioEmbedding(
            model=settings.LOCAL_EMBEDDING_MODEL,
            api_key=settings.LOCAL_API_KEY,
            endpoint=settings.LOCAL_BASE_URL,
        )
    else:
        raise Exception(f"Unknown embedding type: {type}")
