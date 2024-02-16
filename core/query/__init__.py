import pathlib
from typing import List, Union
from langchain_core.vectorstores import VectorStoreRetriever
from core.db import get_vectorstore_from_type
from service.loaders import get_documents
from service.utils.text_split import split_documents
from pathlib import Path
from langchain_community.vectorstores import pinecone, chroma


def get_retriever(search_type: str = "mmr", k: int = 2, **kwargs):
    _path = Path(kwargs.get("doc_path", ""))
    _namespace = kwargs.get("namespace", "default")
    _chunk_size = kwargs.get("chunk_size", 300)
    _chunk_overlap = kwargs.get("chunk_overlap", 0)

    vectorstore = get_vectorstore_from_type(**kwargs).get()

    if not has_namespace(vectorstore, _namespace):
        if not _path.name:
            raise Exception("파일 경로가 입력되지 않았습니다.")

        # docs취득
        kwargs["docs"] = get_docs_from_persist(_path, _chunk_size, _chunk_overlap)
        vectorstore = get_vectorstore_from_type(**kwargs)

    _retriever = VectorStoreRetriever(
        vectorstore=vectorstore,
        search_type=search_type,
        search_kwargs={"k": k},
    )

    return _retriever


def has_namespace(vs: object, namespace):
    if not isinstance(vs, chroma.Chroma):
        return (
            len([_c for _c in vs._client.list_collections() if _c.name == namespace])
            > 0
        )
    elif isinstance(vs, pinecone.Pinecone):
        return namespace in vs._index.describe_index_stats().to_dict()["namespaces"]


def get_docs_from_persist(path: Path, chunk_size: int = 300, chunk_overlap: int = 0):
    _docs = load_docs(path)
    return split_documents(_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def load_docs(path: Union[Path, str]):
    _path = str(path.resolve()) if isinstance(path, Path) else path
    return get_documents(_path)
