from typing import Union, List
from langchain.document_loaders.text import TextLoader
from service.utils.text_split import get_splitter
from cmn.types.loader import LoaderType, Document
from overrides import override


def get_loader(files: Union[List[str], str]):
    return TxtLoader(files)


class TxtLoader(LoaderType):
    docs: List[Document] = []

    def __init__(self, files: Union[List[str], str]):
        self.files = [files] if isinstance(files, str) else files

    @property
    def documents(self) -> List[Document]:
        return self.docs

    @override
    def get_documents(self) -> List[Document]:

        for file in self.files:
            loader = TextLoader(file, encoding="utf-8")
            self.docs.extend(loader.load_and_split(text_splitter=get_splitter()))

        return self.docs
