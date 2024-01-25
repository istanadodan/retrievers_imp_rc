from langchain.docstore.document import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.pinecone import Pinecone
from typing import List
from core.llm import get_embeddings
from langchain.docstore.in_memory import InMemoryDocstore
import logging


def get_vectorstore_from_type(
    *,
    vd_name: str = "chroma",
    docs: List[Document] = None,
    store: object = InMemoryDocstore(),
    index_name: str = None,
    **kwargs,
) -> any:
    _embedding_model = get_embeddings()

    if vd_name == "faiss":
        import faiss

        embedding_word_demension = 768
        _index = faiss.IndexFlatL2(embedding_word_demension)
        return FAISS(
            embedding_function=_embedding_model,
            index=_index,
            docstore=store,
            index_to_docstore_id={},
        )

    if vd_name == "chroma":
        if docs:
            return Chroma.from_documents(documents=docs, embedding=_embedding_model)

        return Chroma(collection_name="pdf_docs", embedding_function=_embedding_model)

    """filename,"""
    if vd_name == "pinecone":
        import pinecone
        import os

        namesapce = kwargs.get("filename")
        if not namesapce:
            raise Exception("filename이 입력되지 않았습니다")

        pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        # try:
        #     return Pinecone.from_existing_index(
        #         index_name=index_name, embedding=_embedding_model, namespace=file
        #     )
        # except:

        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        if index_name not in existing_indexes:
            from time import time

            # Create Index
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-west-2"),
            )

            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

        if docs:
            # index_name은 필수값
            # embedding_chunk_size for llm rateLimit, batch_size for Pinecone to avoid its trafiic
            return Pinecone.from_documents(
                documents=docs,
                embedding=_embedding_model,
                index_name=index_name,
                namespace=namesapce,
            )

        return Pinecone.from_existing_index(
            index_name=index_name, embedding=_embedding_model, namespace=namesapce
        )

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


class VectorStoreSelect:
    def __init__(self, name: str = "faiss") -> None:
        self.name = name

    def get(self, **kwargs):
        _vstore = getattr(self, "get_" + self.name)
        return _vstore(**kwargs)

    def get_faiss(self, embedding_model: str, store: object):
        import faiss

        embedding_word_demension = 768
        _index = faiss.IndexFlatL2(embedding_word_demension)
        return FAISS(
            embedding_function=embedding_model,
            index=_index,
            docstore=store,
            index_to_docstore_id={},
        )

    def get_chroma(self):
        return "chroma"

    def get_pinecone(self):
        from pinecone import Pinecone
        import os

        pc = Pinecone(api_key=os.environ("PINECONE_API_KEY"))
        return pc
