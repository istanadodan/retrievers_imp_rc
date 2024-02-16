from typing import Union, List
from langchain.document_loaders.pdf import PyPDFLoader
from cmn.types.loader import LoaderType, Document
from overrides import override


def get_loader(files: Union[List[str], str]):
    return PDFLoader(files)


class PDFLoader(LoaderType):
    docs: List[Document] = []

    def __init__(self, files: Union[List[str], str]):
        self.files = [files] if isinstance(files, str) else files

    @property
    def documents(self) -> List[Document]:
        return self.docs

    @override
    def get_documents(self) -> List[Document]:

        for file in self.files:
            loader = PyPDFLoader(file)
            self.docs.extend(loader.load())

        return self.docs
