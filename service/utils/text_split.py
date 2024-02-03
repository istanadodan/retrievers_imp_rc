from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_child_splitter(chunk_size: int = 200):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)


def get_parent_splitter(chunk_size: int = 800):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)


def split_documents(
    docs: List[Document], chunk_size: int = 300, chunk_overlap: int = 0
):
    if not docs:
        return []
    _splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return _splitter.split_documents(docs)
