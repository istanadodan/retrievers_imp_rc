from typing import Union, List
from langchain.document_loaders.pdf import PyPDFLoader
from backend.cmn.types.loader import LoaderType, Document
from overrides import override


class PDFLoader(LoaderType):
    @override
    def get_documents(self) -> List[Document]:

        for file in self.files:
            loader = PyPDFLoader(file)
            self.docs.extend(loader.load_and_split(text_splitter=self.splitter))

        return self.docs
