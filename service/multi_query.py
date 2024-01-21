from typing import List
from langchain.retrievers import MultiQueryRetriever
import logging
from core.llm import get_llm
from core.query import get_retriever

logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


def query(query: str, doc_path: str = "./public/R2304581.pdf"):
    import re

    _retriever = get_retriever(k=1, path=doc_path)

    mul_retriever: MultiQueryRetriever = MultiQueryRetriever.from_llm(
        retriever=_retriever, llm=get_llm(), include_original=True
    )
    _result = mul_retriever.get_relevant_documents(query)

    logging.info(f"retrieve output:\n{_result}")
    return list(map(lambda x: re.sub(r"\n+", "\n", x.page_content), _result))
