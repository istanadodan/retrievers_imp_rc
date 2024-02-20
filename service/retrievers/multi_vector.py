"""
    검색된 노드와 연관된 노드들을 찾는다.
    검색된 노드의  상위 노드에 딸린 자식노드가 연관성이 가장 높다고 가정한다.
    적정량의 child/parent node chunking크기를 고려한다. 예에서는 기본 (200/800)
"""

from typing import List
from langchain.retrievers import MultiVectorRetriever
from core.db import get_vectorstore_from_type
from service.utils.text_split import get_splitter
from langchain.storage import InMemoryStore
import logging


def query(query: str, doc_path: str, k: int = 3):
    # vectorstore에 문서를 넣으면, 이 값이 parent값이 되어 버린다.
    # vectorstore = get_vectorstore_from_type(vd_name="chroma")
    # sub_docs = vectorstore.similarity_search(query=query, k=1)

    retriever = multivector_retriever(doc_path, k=k)
    # logging.info(f"store: {len(list(store.yield_keys()))}")

    _result = retriever.get_relevant_documents(query)
    logging.info(f"retrieve output:\n{_result}")

    # return list(map(lambda x: x.page_content, _result))
    return _result


def multivector_retriever(doc_path: str, k: int = 3):
    from service.utils.retrieve_params import get_default_vsparams

    kwargs = get_default_vsparams(doc_path=doc_path)
    vsclient = get_vectorstore_from_type(**kwargs)

    # 외부 DB로부터 doc를 읽어온다.
    # 1. original docs, 2. summary_docs
    store = InMemoryStore()

    retriever = MultiVectorRetriever(
        vectorstore=vsclient.get(),
        docstore=store,
        search_type="mmr",
        search_kwargs={"k": k},
        kwargs={"verbose": True},
        id_key="doc_id",
    )

    return retriever
