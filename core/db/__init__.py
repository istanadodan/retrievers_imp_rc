from langchain.docstore.document import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from typing import List
from core.llm import get_embeddings
from langchain.docstore.in_memory import InMemoryDocstore

def _get_vectorstor_from_documents(docs: List[Document], embedding_model: any):
    return Chroma.from_documents(documents=docs, embedding=embedding_model)


def get_vectorstore(docs: List[Document] = None):
    _embedding_model = get_embeddings()
    if docs:
        return _get_vectorstor_from_documents(docs, _embedding_model)
    return Chroma(collection_name="pdf_docs", embedding_function=_embedding_model)


def get_vectorstore_from_type(
    *,
    vd_name: str = "chroma",
    docs: List[Document] = None,
    store: object = InMemoryDocstore(),
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
        return get_vectorstore(docs)

    else:
        raise FileNotFoundError(f"bad vd_name: {vd_name}")
