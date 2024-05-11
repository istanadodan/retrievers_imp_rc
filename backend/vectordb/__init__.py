from typing import List, Union
from backend.cmn.types.vectorstore import VectoreStoreInf, VectoreStoreMixin, VectoreStoreType
from .vectorstore import *
from pathlib import Path

# def get_retriever(
#     vs_client: VectoreStoreInf, search_type: str = "mmr", k: int = 2, **kwargs
# ):
#     _retriever = VectorStoreRetriever(
#         vectorstore=vs_client.get(),
#         search_type=search_type,
#         search_kwargs={"k": k},
#     )

#     return _retriever


# 벡터DB명을 입력받아 객체생성
def get_vectorstore_from_type(
    vd_name: str,
    **kwargs,
) -> Union[VectoreStoreMixin, VectoreStoreInf]:
    from core.llm import get_embeddings

    if vd_name not in get_available_vectorstores():
        raise ValueError(
            "vectorstores are not installed. Please install them with `pip install vectorstores`"
        )
    _client = select_vectorstore(vd_name, get_embeddings(), **kwargs)
    _client.create()

    return _client

    # vectors = [{"id": "vec1", "values": [0.1, 0.1]}]

    # # When upserting larger amounts of data, upsert data in batches of 100-500 vectors over multiple upsert requests.
    # index.upsert(vectors=vectors, namespace="filename")
    # logging.info(f"upsert stat: {index.describe_index_stats()}")

    # # perform a query for the index vector
    # index.query(namespace="filename", vector=[0.3], top_k=3, include_values=True)
    # # delete a index
    # pc.delete_index("db_name")
    # return pc


def get_available_vectorstores() -> list[str]:
    """사용 가능한 vectorstore 목록을 반환한다."""

    return VectoreStoreType.values()


def select_vectorstore(
    vectorstore_type: str,
    embedding_model: object,
    **kwargs,
) -> VectoreStoreInf:
    """타입을 입력받아, vectorstore wrapper를 생성하여 반환한다."""

    _persist_dir = kwargs.get("persist_dir") or str(Path(f'./core/db/{vectorstore_type}').resolve())

    if vectorstore_type == VectoreStoreType.FAISS:
        kwargs = {
            "embedding_model": embedding_model,
            "persist_dir": _persist_dir,
            "index_name": kwargs.get("namespace"),  # 문서
            "dim": 769,
        }
        return FaissVs(**kwargs)

    elif vectorstore_type == VectoreStoreType.CHROMA:
        kwargs = {
            "embedding_model": embedding_model,
            "persist_dir": _persist_dir,
            "collection_name": kwargs.get("namespace"),  # 문서
        }

        return ChromaVs(**kwargs)

    elif vectorstore_type == VectoreStoreType.PINECONE:
        kwargs = {
            "embedding_model": embedding_model,
            "index_name": kwargs.get("index_name"),  # 문서그룹
            "namespace": kwargs.get("namespace"),  # 문서
        }
        if not kwargs["index_name"]:
            raise ValueError("index_name is required")
        return PineconeVs(**kwargs)

    else:
        raise ValueError(f"vectorstore_type: {vectorstore_type} is not supported")
