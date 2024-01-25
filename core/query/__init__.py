import pathlib
from typing import List, Union
from langchain_core.vectorstores import VectorStoreRetriever
from core.db import get_vectorstore_from_type
from service.loader.pdf_loader import get_documents
from service.utils.text_split import split_documents


def get_retriever(search_type: str = "mmr", k: int = 2, **kwargs):
    _retriever = VectorStoreRetriever(
        vectorstore=get_vectorstore_from_type(**kwargs),
        search_type=search_type,
        search_kwargs={"k": k},
    )

    if False and "path" in kwargs:
        _docs = load_docs(kwargs["path"])
        _chunk_size = kwargs.get("chunk_size", 300)
        _chunk_overlap = kwargs.get("chunk_overlap", 0)
        _retriever.add_documents(
            split_documents(
                _docs, chunk_size=_chunk_size, chunk_overlap=_chunk_overlap
            ),
            index_name=kwargs.get("index_name"),
            namespace=kwargs.get("filename")
            # **kwargs
        )

    return _retriever


def load_docs(path: Union[pathlib.Path, str]):
    path = pathlib.Path(path) if isinstance(path, str) else path
    return get_documents(str(path.resolve()))
