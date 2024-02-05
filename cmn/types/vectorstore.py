from typing import List
from langchain.docstore.document import Document
from langchain.vectorstores import VectorStore
from abc import ABC, abstractmethod


class VectoreStoreInf(ABC):
    client: object = None
    vectorstore: object = None

    @abstractmethod
    def get(self, **kwargs) -> VectorStore:
        """"""

    def add(self, docs: List[Document]):
        """"""
        self.vectorstore.add_documents(docs)

    @abstractmethod
    def search(self, query: str, **kwargs):
        """"""

    @abstractmethod
    def delete(self, id: any, **kwargs):
        """"""

    def save(self, **kwargs):
        """"""
