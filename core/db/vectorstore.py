from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.pinecone import Pinecone
from typing import List


class VectorStoreSelect:
    def __init__(self, name: str = "faiss") -> None:
        self.name = name

    def get(self, **kwargs):
        _vstore = getattr(self, "get_" + self.name)
        return _vstore(**kwargs)

    def get_faiss(self, embedding_model: str, store: object):
        import faiss

        embedding_word_demension = 768
        _index = faiss.IndexFlatL2(embedding_word_demension)
        return FAISS(
            embedding_function=embedding_model,
            index=_index,
            docstore=store,
            index_to_docstore_id={},
        )

    def get_chroma(self):
        return "chroma"

    def get_pinecone(self):
        from pinecone import Pinecone
        import os

        pc = Pinecone(api_key=os.environ("PINECONE_API_KEY"))
        return pc
