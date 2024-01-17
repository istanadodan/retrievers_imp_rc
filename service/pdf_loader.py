from typing import List
from typing import Union
from langchain.document_loaders.pdf import PyPDFLoader

def get_documents(files:Union[List[str], str]):
    files = files if isinstance(files, list) else [files]
    docs = []
    for file in files:
        docs.extend(PyPDFLoader(file).load())

    return docs
