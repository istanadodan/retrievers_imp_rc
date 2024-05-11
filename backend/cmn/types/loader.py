from typing import List, Union
from langchain.docstore.document import Document
from abc import ABC, abstractmethod


class LoaderType(ABC):
    docs: List[Document]

    def __init__(self, files: Union[List[str], str], splitter: object):
        self.files = [files] if isinstance(files, str) else files
        self.splitter = splitter
        self.docs = []

    @property
    def documents(self) -> List[Document]:
        return self.docs

    @abstractmethod
    def get_documents(self) -> List[Document]:
        """문서를 파싱하고 페이지별로 나눠 반환한다."""
