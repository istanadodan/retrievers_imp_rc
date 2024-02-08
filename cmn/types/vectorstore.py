from typing import List
from langchain.docstore.document import Document
from langchain.vectorstores import VectorStore
from abc import ABC, abstractmethod


class VectoreStoreInf(ABC):
    client: object = None
    vectorstore: object = None

    @abstractmethod
    def create(self, **kwargs) -> VectorStore:
        """"""

    def get(self) -> VectorStore:
        """"""
        return self.vectorstore

    def add(self, docs: List[Document], **kwargs):
        """"""
        self.vectorstore.add_documents(docs)

    @abstractmethod
    def search(self, query: str, **kwargs):
        """"""

    @abstractmethod
    def delete(self, id: any, **kwargs):
        """"""
