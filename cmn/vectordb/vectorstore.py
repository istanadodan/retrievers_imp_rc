from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from typing import List
from cmn.types.vectorstore import VectoreStoreInf, VectoreStoreMixin
from pathlib import Path

import os


# def get_available_vectorstores() -> list[str]:
#     """사용 가능한 vectorstore 목록을 반환한다."""

#     return VectoreStoreType.values()


# def select_vectorstore(
#     vectorstore_type: str,
#     embedding_model: object,
#     **kwargs,
# ) -> VectoreStoreInf:
#     """타입을 입력받아, vectorstore wrapper를 생성하여 반환한다."""

#     if vectorstore_type == VectoreStoreType.FAISS:
#         kwargs = {
#             "embedding_model": embedding_model,
#             "persist_dir": kwargs.get("persist_dir"),
#             "index_name": kwargs.get("namespace"),  # 문서
#             "dim": 769,
#         }
#         return FaissVs(**kwargs)

#     elif vectorstore_type == VectoreStoreType.CHROMA:
#         kwargs = {
#             "embedding_model": embedding_model,
#             "persist_dir": kwargs.get("persist_dir"),
#             "collection_name": kwargs.get("namespace"),  # 문서
#         }

#         return ChromaVs(**kwargs)

#     elif vectorstore_type == VectoreStoreType.PINECONE:
#         kwargs = {
#             "embedding_model": embedding_model,
#             "index_name": kwargs.get("index_name"),  # 문서그룹
#             "namespace": kwargs.get("namespace"),  # 문서
#         }
#         if not kwargs["index_name"]:
#             raise ValueError("index_name is required")
#         return PineconeVs(**kwargs)

#     else:
#         raise ValueError(f"vectorstore_type: {vectorstore_type} is not supported")


class PineconeVs(VectoreStoreMixin, VectoreStoreInf):

    def __init__(self, index_name: str, embedding_model: object) -> None:
        import pinecone

        self.index_name = index_name
        self.embedding_model = embedding_model
        # 초기화
        self.client = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.spec = pinecone.ServerlessSpec(cloud="aws", region="us-west-2")
        # 인덱스 존재 여부를 확인하고 없으면 생성.
        self._check_and_create_index()

    def create(self, namesapce: str):
        from langchain_community.vectorstores.pinecone import Pinecone

        self.vectorstore = Pinecone.from_existing_index(
            index_name=self.index_name,
            namespace=namesapce,
            embedding=self.embedding_model,
        )
        return self.vectorstore

    def delete(self, id: any):
        return self.vectorstore.delete(namespace=id, delete_all=True)

    def _check_and_create_index(self):
        """인덱스명은 db명"""
        existing_indexes = [
            index_info["name"] for index_info in self.client.list_indexes()
        ]
        if self.index_name not in existing_indexes:
            from time import time

            # Create Index
            self.client.create_index(
                name=self.index_name, dimension=768, metric="cosine", spec=self.spec
            )
            start_time = time()
            while not self.client.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
                # 현재부터 10초가 지났는지를 체크한다.
                if time() - start_time > 10:
                    raise Exception("Timeout waiting for index to be ready")


class FaissVs(VectoreStoreMixin, VectoreStoreInf):

    def __init__(
        self,
        embedding_model: object,
        index_name: str,
        persist_dir: str,
        dim: int = 768,
    ) -> None:
        import faiss

        self.embedding_model = embedding_model
        self.persist_dir = persist_dir
        self.client = faiss.IndexFlatL2(dim)
        self.index_name = index_name

    def create(self):

        os.makedirs(self.persist_dir, exist_ok=True)

        try:
            self.vectorstore = FAISS.load_local(
                index_name=self.index_name,
                folder_path=self.persist_dir,
                embeddings=self.embedding_model,
            )
        except Exception as e:
            if not self.store:
                raise Exception("store가 None입니다.")

            self.vectorstore = FAISS(
                embedding_function=self.embedding_model,
                index=self.client,
                docstore=self.store,
                index_to_docstore_id={},
            )
        return self.vectorstore

    def delete(self):
        """인덱스파일을 삭제한다."""
        try:
            os.remove(self.persist_dir)
        except Exception as e:
            """"""

    def add(self, docs, **kwargs):
        super().add(docs)
        namespace = kwargs.get("namespace")
        db_name = kwargs.get("vd_name")
        persist_dir = Path(os.getcwd()) / "core" / "db" / db_name
        self.vectorstore.save_local(self, folder_path=persist_dir, index_name=namespace)

    def exists(self, name: str) -> bool:
        """테이블 혹은 컬렉션이 DB에 존재하는지 여부 반환"""
        return self.vectorstore != None


class ChromaVs(VectoreStoreMixin, VectoreStoreInf):

    def __init__(
        self, embedding_model: object, collection_name: str, persist_dir: str
    ) -> None:
        self.embedding_model = embedding_model
        self.persist_dir = persist_dir
        self.collection_name = collection_name

    def create(self):
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_dir,
        )

        return self.vectorstore

    def delete(self):
        """현재 collection을 삭제한다."""
        self.get().delete_collection()

    def exists(self, name: str) -> bool:
        self.has_index = (
            len(
                list(
                    filter(
                        lambda x: x.name == name,
                        self.vectorstore._client.list_collections(),
                    )
                )
            )
            > 0
        )
        return self.has_index
