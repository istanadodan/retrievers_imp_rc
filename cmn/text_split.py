from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_splitter(chunk_size: int = 200, chunk_overlap=0):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


def split_documents(
    docs: List[Document], chunk_size: int = 300, chunk_overlap: int = 0
):
    if not docs:
        return []
    return get_splitter(chunk_size, chunk_overlap).split_documents(docs)
