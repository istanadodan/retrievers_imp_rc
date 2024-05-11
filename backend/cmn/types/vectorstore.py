from enum import Enum
from typing import List
from langchain.docstore.document import Document
from langchain.vectorstores import VectorStore
from abc import ABC, abstractmethod


class VectoreStoreType(Enum):
    FAISS = "faiss"
    CHROMA = "chroma"
    PINECONE = "pinecone"

    @classmethod
    def values(cls):
        output = []
        for _, member in cls.__members__.items():
            output.append(member.value)
        return output

    def __eq__(self, other):
        if isinstance(other, VectoreStoreType):
            return self.value == other
        return self.value == other


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

    def search(self, query: str, **kwargs: any) -> List[Document]:
        """질의와 질의방식을 입력받아 검색한다."""
        search_type = kwargs.get("search_type")
        if not search_type:
            search_type = "similarity"
        else:
            kwargs.pop("search_type")
        return self.get().search(query, search_type, **kwargs)

    @abstractmethod
    def delete(self, **kwargs: any):
        """"""

    def save(self, docs: List[Document], **kwargs):
        """벡터db에 저장한다."""
        self.add(docs, **kwargs)
