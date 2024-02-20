from typing import List
from langchain.docstore.document import Document
from langchain.vectorstores import VectorStore
from abc import ABC, abstractmethod


class VectoreStoreMixin:
    has_index: bool = False

    @property
    def bindex(self) -> bool:
        return self.has_index


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
    def exists(self, name: str) -> bool:
        """테이블 혹은 컬렉션이 DB에 존재하는지 여부 반환"""

    @abstractmethod
    def search(self, query: str, **kwargs):
        """"""

    @abstractmethod
    def delete(self, id: any, **kwargs):
        """"""

    def save(self, docs: List[Document], **kwargs):
        """벡터db에 저장한다."""
        self.add(docs, **kwargs)
