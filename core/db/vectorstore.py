from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from typing import List
from cmn.types.vectorstore import VectoreStoreInf
import os


class PineconeVs(VectoreStoreInf):

    def __init__(self, index_name: str, embedding_model: object) -> None:
        import pinecone

        self.index_name = index_name
        self.embedding_model = embedding_model
        # 초기화
        self.client = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.spec = pinecone.ServerlessSpec(cloud="aws", region="us-west-2")
        # 인덱스 존재 여부를 확인하고 없으면 생성.
        self._check_index()

    def get(self, namesapce: str, docs: List[any] = None):
        from langchain_community.vectorstores.pinecone import Pinecone

        if docs:
            self.vectorstore = Pinecone.from_documents(
                documents=docs,
                embedding=self.embedding_model,
                namesapce=namesapce,
                index_name=self.index_name,
            )
        else:
            self.vectorstore = Pinecone.from_existing_index(
                index_name=self.index_name,
                namespace=namesapce,
                embedding=self.embedding_model,
            )
        return self.vectorstore

    def _check_index(self):
        """인덱스명은 db명"""
        existing_indexes = [
            index_info["name"] for index_info in self.client.list_indexes()
        ]
        if self.index_name not in existing_indexes:
            from time import time

            # Create Index
            self.client.create_index(
                name=self.index_name, dimension=768, metric="cosine", spec=self.spec
            )

            while not self.client.describe_index(self.index_name).status["ready"]:
                time.sleep(1)

    def delete(self, id: any, **kwargs):
        return self.vectorstore.delete(namespace=id, delete_all=True)

    def search(self, query: str, **kwargs):
        """"""

    def delete(self, id: any, **kwargs):
        """"""
        return super().delete(id, **kwargs)


class FaissVs(VectoreStoreInf):

    def __init__(self, embedding_model: object, dim: int = 768) -> None:
        import faiss

        self.embedding_model = embedding_model
        self.client = faiss.IndexFlatL2(embedding_word_demension=dim)

    def get(self, store: object = None):

        self.vectorstore = FAISS(
            embedding_function=self.embedding_model,
            index=self.client,
            docstore=store,
            index_to_docstore_id={},
        )
        return self.vectorstore

    def search(self, query: str, **kwargs):
        """"""

    def delete(self, id: any, **kwargs):
        """"""


class ChromaVs(VectoreStoreInf):

    def __init__(self, embedding_model: object) -> None:
        import faiss

        self.embedding_model = embedding_model

    def get(self, collection_name: str, docs: List[any] = None):

        if docs:
            self.vectorstore = Chroma.from_documents(
                collection_name=collection_name,
                documents=docs,
                embedding=self.embedding_model,
            )
        else:
            self.vectorstore = Chroma(
                collection_name=collection_name, embedding_function=self.embedding_model
            )
        return self.vectorstore

    def search(self, query: str, **kwargs):
        """"""

    def delete(self, id: any, **kwargs):
        """"""
