from typing import List
from langchain.docstore.document import Document
from langchain.vectorstores import VectorStore
from abc import ABC


class VectoreStoreInf(ABC):
    def get(self, **kwargs) -> VectorStore:
        """"""

    def add(self, docs: List[Document]):
        """"""

    def search(self, query: str, **kwargs):
        """"""

    def delete(self, id: any, **kwargs):
        """"""
