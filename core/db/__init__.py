from typing import List, Union
from cmn.types.vectorstore import VectoreStoreInf, VectoreStoreMixin
from core.llm import get_embeddings
from langchain.docstore.in_memory import InMemoryDocstore
from pathlib import Path


"""생성자 혹은 from_documents를 통해 vectorstore 생성
"""


def get_vectorstore_from_type(
    vd_name: str,
    **kwargs,
) -> Union[VectoreStoreMixin, VectoreStoreInf]:
    from .vectorstore import select_vectorstore, get_available_vectorstores

    if vd_name not in get_available_vectorstores():
        raise ValueError(
            "vectorstores are not installed. Please install them with `pip install vectorstores`"
        )
    _vs_wapper = select_vectorstore(vd_name, get_embeddings(), **kwargs)
    _vs_wapper.create()

    return _vs_wapper

    # vectors = [{"id": "vec1", "values": [0.1, 0.1]}]

    # # When upserting larger amounts of data, upsert data in batches of 100-500 vectors over multiple upsert requests.
    # index.upsert(vectors=vectors, namespace="filename")
    # logging.info(f"upsert stat: {index.describe_index_stats()}")

    # # perform a query for the index vector
    # index.query(namespace="filename", vector=[0.3], top_k=3, include_values=True)
    # # delete a index
    # pc.delete_index("db_name")
    # return pc
