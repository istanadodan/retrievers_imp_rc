from typing import List
from langchain.retrievers import MultiQueryRetriever
import logging
from core.llm import get_llm
from core.query import get_retriever
import re

logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


def query(query: str, doc_path: str, k: int = 1):
    mul_retriever: MultiQueryRetriever = mquery_retriever(doc_path, k)
    _result = mul_retriever.get_relevant_documents(query)

    logging.info(f"retrieve output:\n{_result}")
    return list(map(lambda x: re.sub(r"\n+", "\n", x.page_content), _result))


def mquery_retriever(doc_path: str, k: int = 1):
    _retriever = get_retriever(k=k, path=doc_path)
    if not _retriever:
        return

    return MultiQueryRetriever.from_llm(
        retriever=_retriever, llm=get_llm(), include_original=False
    )
