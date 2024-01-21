"""
    검색된 노드와 연관된 노드들을 찾는다.
    검색된 노드의  상위 노드에 딸린 자식노드가 연관성이 가장 높다고 가정한다.
    적정량의 child/parent node chunking크기를 고려한다. 예에서는 기본 (200/800)
"""
import pathlib
from typing import List
from langchain.docstore.document import Document
from langchain.retrievers import ParentDocumentRetriever
import logging


def query(query: str):
    from core.db import get_vectorstore
    from service.loader import pdf_loader
    from service.utils.text_split import get_child_splitter, get_parent_splitter
    from langchain.storage import InMemoryStore

    # vectorstore에 문서를 넣으면, 이 값이 parent값이 되어 버린다.
    vectorstore = get_vectorstore()
    # 내부 docstore 작성 시, 저장메모리
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        parent_splitter=get_parent_splitter(),
        child_splitter=get_child_splitter(),
        search_kwargs={"k": 3},
    )
    # child splitter를 사용한 문서를 vectorstore에 업데이트 한다.
    docs: List[Document] = pdf_loader.get_documents(
        str(pathlib.Path("./public/R2304581.pdf").resolve())
    )

    retriever.add_documents(documents=docs, ids=None)
    logging.info(f"store: {len(list(store.yield_keys()))}")
    # sub_docs = vectorstore.similarity_search(query=query, k=1)

    _result = retriever.get_relevant_documents(query)
    logging.info(f"retrieve output:\n{_result}")

    return list(map(lambda x: x.page_content, _result))
