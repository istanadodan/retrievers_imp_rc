"""
    검색된 노드와 연관된 노드들을 찾는다.
    검색된 노드의  상위 노드에 딸린 자식노드가 연관성이 가장 높다고 가정한다.
    적정량의 child/parent node chunking크기를 고려한다. 예에서는 기본 (200/800)
"""

import os
from pathlib import Path
from typing import List
from langchain.docstore.document import Document
from langchain.retrievers import ParentDocumentRetriever
from core.db import get_vectorstore_from_type
from service.loader import pdf_loader
from service.utils.text_split import get_splitter
from langchain.storage import InMemoryStore
import logging


def query(query: str, doc_path: str, k: int = 3):
    # vectorstore에 문서를 넣으면, 이 값이 parent값이 되어 버린다.
    # vectorstore = get_vectorstore_from_type(vd_name="chroma")
    # sub_docs = vectorstore.similarity_search(query=query, k=1)

    # 내부 docstore 작성 시, 저장메모리
    store = InMemoryStore()

    retriever = pdoc_retriever(doc_path, k=k)
    logging.info(f"store: {len(list(store.yield_keys()))}")

    _result = retriever.get_relevant_documents(query)
    logging.info(f"retrieve output:\n{_result}")

    # return list(map(lambda x: x.page_content, _result))
    return _result


def pdoc_retriever(doc_path: str, k: int = 3):
    # child splitter를 사용한 문서를 vectorstore에 업데이트 한다.
    docs: List[Document] = pdf_loader.get_documents(doc_path)
    if not docs:
        return

    # vectorstore에 문서를 넣으면, 이 값이 parent doc이 되어 버린다.
    from service.utils.retrieve_params import get_default_vsparams

    kwargs = get_default_vsparams(doc_path=doc_path, vd_name="chroma")
    _vs_wrapper = get_vectorstore_from_type(**kwargs)
    # 내부 docstore 작성 시, cache 메모리
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=_vs_wrapper.get(),
        docstore=store,
        parent_splitter=get_splitter(350),
        child_splitter=get_splitter(100),
        search_kwargs={"k": k},
        kwargs={"verbose": True},
    )
    # parent/child docstore 생성
    retriever.add_documents(documents=docs, ids=None)

    return retriever
