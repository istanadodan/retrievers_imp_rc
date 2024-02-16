from typing import List
from langchain.docstore.document import Document
from abc import ABC, abstractmethod


class LoaderType(ABC):
    @abstractmethod
    def get_documents(self) -> List[Document]:
        """문서를 파싱하고 페이지별로 나눠 반환한다."""
