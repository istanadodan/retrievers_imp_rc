from typing import Union, List
from langchain_community.document_loaders import WebBaseLoader
from service.utils.text_split import get_splitter
from cmn.types.loader import LoaderType, Document
from overrides import override


def get_loader(files: Union[List[str], str]):
    return WebLoader(files)


class WebLoader(LoaderType):
    @override
    def get_documents(self) -> List[Document]:
        def filter_func(doc: Document):
            cond1 = "." in doc.page_content
            cond2 = len(doc.page_content) > 20
            return cond1 and cond2

        def map_func(doc: Document):
            doc.metadata = {"source": "web"}
            doc.page_content = " ".join(doc.page_content.split("\n")[-3:])
            return doc

        for url in self.urls:
            loader = WebBaseLoader(url)
            docs = loader.load_and_split(text_splitter=self.splitter)
            self.docs.extend(list(filter(filter_func, map(map_func, docs))))
            pass

        return self.docs
