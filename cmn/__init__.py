from .loaders import get_documents_from_file
from cmn.text_split import split_documents
from pathlib import Path
from langchain_community.vectorstores import pinecone, chroma


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
        get_documents_from_file(str(path.resolve())),
        chunk_size=kwargs.get("chunk_size", 300),
        chunk_overlap=kwargs.get("chunk_overlap", 0),
    )
