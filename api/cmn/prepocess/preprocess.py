from enum import Enum, auto
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


class RagFileType(Enum):
    PDF = auto()
    TXT = auto()
    MS = auto()
    msx = auto
    hwp = auto


class DefaultPreProcessor:
    """문서 청킹처리"""

    def __init__(self, embed_model: object, chunk_params: dict):
        """임베딩모델, 파일경로, 파일타입"""
        self.embed_model = embed_model
        self.check_params = chunk_params

    def load_document(self, file_type: RagFileType, file_path: Path, **kwargs) -> list[Document]:
        if file_type == RagFileType.PDF:
            loader = PyMuPDFLoader(file_path=file_path.resolve())
        else:
            raise Exception("not support file type")
        
        return loader.load()

    def document_chunking(
        self,
        documents: list[Document],
        split_type: str = "recursive",
    ) -> list[Document]:
        if split_type == "recursive":
            spliter = RecursiveCharacterTextSplitter(**self.check_params)
            split_docs = spliter.split_documents(documents)
        return split_docs
    
    def make_embeddings(self, chunks:list[Document], batch_size:int=32, **kwargs) -> list[list[float]]:
        output = []
        for index in range(0, len(chunks),batch_size):
            batch = chunks[index : index + batch_size]
            _emb = self.embed_model.embed_documents(batch)
            output.extend(_emb)
        return output

    def process(self, **kwargs) -> list[Document]:
        docs = self.load_document(**kwargs)
        split_docs = self.document_chunking(documents=docs)
        vectors = self.make_embeddings(chunks=split_docs, **kwargs)
        return dict(docs=split_docs, vectors=vectors)
