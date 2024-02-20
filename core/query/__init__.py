import pathlib
from typing import List, Union
from langchain_core.vectorstores import VectorStoreRetriever
from cmn.types.vectorstore import VectoreStoreInf
from core.db import get_vectorstore_from_type
from service.loaders import get_documents
from service.utils.text_split import split_documents
from pathlib import Path
from langchain_community.vectorstores import pinecone, chroma


def get_retriever(
    vs_client: VectoreStoreInf, search_type: str = "mmr", k: int = 2, **kwargs
):
    _retriever = VectorStoreRetriever(
        vectorstore=vs_client.get(),
        search_type=search_type,
        search_kwargs={"k": k},
    )

    return _retriever


# def has_namespace(vs: object, namespace):
#     if isinstance(vs, chroma.Chroma):
#         return (
#             len([_c for _c in vs._client.list_collections() if _c.name == namespace])
#             > 0
#         )
#     elif isinstance(vs, pinecone.Pinecone):
#         return namespace in vs._index.describe_index_stats().to_dict()["namespaces"]


def get_docs_from_persist(path: Path, **kwargs):
    return split_documents(
        get_documents(str(path.resolve())),
        chunk_size=kwargs.get("chunk_size", 300),
        chunk_overlap=kwargs.get("chunk_overlap", 0),
    )
