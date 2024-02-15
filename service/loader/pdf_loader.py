from typing import Union, List
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.text import TextLoader


class Loader:
    def __init__(self, file: str):
        _type = file.split(".")[-1].lower()
        if _type == "pdf":
            self.loader = PyPDFLoader(file)
        elif _type == "txt":
            self.loader = TextLoader(file, encoding="utf-8")

    def load(self):
        if not hasattr(self, "loader"):
            return []
        if hasattr(self.loader, "load_and_split"):
            return self.loader.load_and_split()
        else:
            return self.loader.load()


def get_documents(files: Union[List[str], str], **kwargs):
    files = files if isinstance(files, list) else [files]
    docs = []

    for file in files:
        loader = Loader(file)
        docs.extend(loader.load())

    return docs
