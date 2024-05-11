from langchain.retrievers.bm25 import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import logging
from backend.vectordb import get_vectorstore_from_type
from core.llm import get_llm, get_embeddings
import re

from backend.cmn.loaders import get_documents_from_file
from backend.cmn.text_split import split_documents


def query(query: str, doc_path: str, k: int = 1):
    _retriever: EnsembleRetriever = ensembles_retriever(doc_path, k, dense_rate=0.6)
    _result = _retriever.get_relevant_documents(query)

    logging.info(f"retrieve output:\n{_result}")
    return list(map(lambda x: re.sub(r"\n+", "\n", x.page_content), _result))


def ensembles_retriever(doc_path: str, k: int, dense_rate: float = 0.0):
    from backend.retrievers.retriever_default_param import get_default_vsparams

    docs = split_documents(get_documents_from_file(files=doc_path), chunk_size=200)
    _sparse_retriever = BM25Retriever.from_documents(docs, k=k)
    # _sparse_retriever.k = k

    kwargs = get_default_vsparams(doc_path=doc_path)
    vsclient = get_vectorstore_from_type(**kwargs)

    _dense_retriever = vsclient.get().as_retriever(
        search_type="mmr",
        search_kwargs={"k": k},
    )

    if not _dense_retriever or not _sparse_retriever:
        raise Exception("retriever 오류")

    # 앙상블 조회기
    return EnsembleRetriever(
        retrievers=[_sparse_retriever, _dense_retriever],
        weights=[1 - dense_rate, dense_rate],
    )
