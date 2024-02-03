from langchain.docstore.document import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_community.vectorstores import VectorStore
from typing import List
from core.llm import get_embeddings
from langchain.docstore.in_memory import InMemoryDocstore
from .vectorstore import PineconeVs, FaissVs, ChromaVs

"""생성자 혹은 from_documents를 통해 vectorstore 생성
"""


def get_vectorstore_from_type(
    *,
    vd_name: str = "chroma",
    docs: List[Document] = None,
    store: object = InMemoryDocstore(),
    **kwargs,
) -> any:
    _embedding_model = get_embeddings()

    namespace = kwargs.get("namespace")
    index_name = kwargs.get("index_name")

    if vd_name == "faiss":
        _vs_wapper = FaissVs(index_name=index_name, embeddings=_embedding_model)

        return _vs_wapper.get(store=store)

    if vd_name == "chroma":
        _vs_wapper = ChromaVs(embedding_model=_embedding_model)

        return _vs_wapper.get(collection_name=namespace, docs=docs)

    """filename,"""
    if vd_name == "pinecone":
        _vs_wapper = PineconeVs(index_name=index_name, embedding_model=_embedding_model)

        return _vs_wapper.get(namesapce=namespace, docs=docs)

        # vectors = [{"id": "vec1", "values": [0.1, 0.1]}]

        # # When upserting larger amounts of data, upsert data in batches of 100-500 vectors over multiple upsert requests.
        # index.upsert(vectors=vectors, namespace="filename")
        # logging.info(f"upsert stat: {index.describe_index_stats()}")

        # # perform a query for the index vector
        # index.query(namespace="filename", vector=[0.3], top_k=3, include_values=True)
        # # delete a index
        # pc.delete_index("db_name")
        # return pc

    else:
        raise FileNotFoundError(f"bad vd_name: {vd_name}")
