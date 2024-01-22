"""
    LLM으로 하여금 관련된 유사질의를 4개를 만들어 수행.
    질의 갯수는 default promptTemplate에 존재.
"""
from typing import Union
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.retrievers import MultiQueryRetriever
from core.llm import get_llm
from core.query import get_retriever
import logging
from streamlit.runtime.uploaded_file_manager import UploadedFile
from service import multi_query, parent_document
from enum import Enum, auto

logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


class QueryType(Enum):
    Multi_Query = auto()
    Parent_Document = auto()


def query(query: str, path: str, query_type: QueryType = QueryType.Multi_Query):
    _k = 1
    query_type_map = {
        QueryType.Multi_Query: multi_query.mquery_retriever,
        QueryType.Parent_Document: parent_document.pdoc_retriever,
    }

    _retriever = query_type_map.get(query_type, None)(path, k=_k)
    # 문서 포함 retriever
    if not _retriever:
        return {"result": "문서를 준비하지 못했습니다"}

    # 자동 쿼리생성하는 retriever
    r_qa = RetrievalQA.from_llm(
        llm=get_llm(),
        retriever=_retriever,
        return_source_documents=True,
        verbose=True,
    )
    answer = r_qa.invoke({"query": query})
    return {x: answer[x] for x in iter(answer) if x in ["result", "source_documents"]}
