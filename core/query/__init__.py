import pathlib
from typing import List, Union
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from core.db import get_vectorstore_from_type
from service.loader.pdf_loader import get_documents
from pathlib import Path
from service.utils.text_split import split_documents
from langchain.docstore.document import Document


def get_retriever(search_type: str = "mmr", k: int = 2, **kwargs):
    _retriever = VectorStoreRetriever(
        vectorstore=get_vectorstore_from_type(vd_name="faiss"),
        search_type=search_type,
        search_kwargs={"k": k},
    )
    
    if "path" in kwargs:
        if not kwargs.get("path"): return
        
        _docs = load_docs(kwargs.get("path", "./public/R2304581.pdf"))
        _chunk_size = kwargs.get("chunk_size", 300)
        _chunk_overlap = kwargs.get("chunk_overlap", 0)
        _retriever.add_documents(
            split_documents(_docs, chunk_size=_chunk_size, chunk_overlap=_chunk_overlap)
        )

    return _retriever


def load_docs(path: Union[pathlib.Path, str]):
    path = pathlib.Path(path) if isinstance(path, str) else path
    return get_documents(str(path.resolve()))
