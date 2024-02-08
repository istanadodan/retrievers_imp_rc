from langchain.docstore.document import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_community.vectorstores import VectorStore
from typing import List
from cmn.types.vectorstore import VectoreStoreInf
from core.llm import get_embeddings
from langchain.docstore.in_memory import InMemoryDocstore
from .vectorstore import PineconeVs, FaissVs, ChromaVs, select_vectorstore
from pathlib import Path


"""생성자 혹은 from_documents를 통해 vectorstore 생성
"""


def get_vectorstore_from_type(
    vd_name: str,
    docs: List[Document] = None,
    store: object = InMemoryDocstore(),
    **kwargs,
) -> VectoreStoreInf:

    _vs_params = {
        "namespace": kwargs.get("namespace", "default"),
        "index_name": kwargs.get("index_name", "default"),
        "persist_dir": str((Path.cwd() / "core" / "db" / vd_name).resolve()),
    }
    _vs_wapper = select_vectorstore(vd_name, get_embeddings(), **_vs_params)

    if vd_name == "faiss":
        _vs_wapper.create(store=store, docs=docs)

    elif vd_name == "chroma":
        _vs_wapper.create(collection_name=_vs_params["namespace"], docs=docs)

    elif vd_name == "pinecone":
        _vs_wapper.create(namesapce=_vs_params["namespace"], docs=docs)

    else:
        raise FileNotFoundError(f"bad vd_name: {vd_name}")

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
