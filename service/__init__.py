"""
    LLM으로 하여금 관련된 유사질의를 4개를 만들어 수행.
    질의 갯수는 default promptTemplate에 존재.
"""
from langchain.chains.retrieval_qa.base import RetrievalQA
from core.llm import get_llm
from service import multi_query, parent_document, contextual_compression
from enum import Enum, auto
import logging

logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


class QueryType(Enum):
    Multi_Query = auto()
    Contextual_Compression = auto()
    Parent_Document = auto()


def query(query: str, path: str, query_type: QueryType):
    _k = 3
    query_type_map = {
        QueryType.Multi_Query: multi_query.mquery_retriever,
        QueryType.Contextual_Compression: contextual_compression.compression_retriever,
        QueryType.Parent_Document: parent_document.pdoc_retriever,
    }

    _retriever = query_type_map[query_type](path, k=_k)
    # 문서 포함 retriever
    if not _retriever:
        return {"result": "문서를 준비하지 못했습니다"}

    # 자동 쿼리생성하는 retriever
    r_qa = RetrievalQA.from_llm(
        llm=get_llm(),
        retriever=_retriever,
        return_source_documents=True,
        verbose=True,
        llm_chain_kwargs={"verbose": True},
    )
    answer = r_qa.invoke({"query": query})
    return {x: answer[x] for x in iter(answer) if x in ["result", "source_documents"]}
