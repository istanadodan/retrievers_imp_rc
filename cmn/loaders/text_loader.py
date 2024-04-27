from typing import Union, List
from langchain.document_loaders.text import TextLoader
from cmn.text_split import get_splitter
from cmn.types.loader import LoaderType, Document
from overrides import override


class TxtLoader(LoaderType):
    @override
    def get_documents(self) -> List[Document]:

        for file in self.files:
            loader = TextLoader(file, encoding="utf-8")
            self.docs.extend(loader.load_and_split(text_splitter=self.splitter))

        return self.docs
