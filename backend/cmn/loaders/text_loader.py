from typing import Union, List
from langchain_community.document_loaders import TextLoader
from backend.cmn.types.loader import LoaderType, Document
from overrides import override


class TxtLoader(LoaderType):
    @override
    def get_documents(self) -> List[Document]:

        for file in self.files:
            loader = TextLoader(file, encoding="utf-8")
            self.docs.extend(loader.load_and_split(text_splitter=self.splitter))

        return self.docs
