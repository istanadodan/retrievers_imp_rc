from pathlib import Path
from typing import List, Union
from langchain.retrievers import MultiQueryRetriever
import logging
from core.db import get_vectorstore_from_type
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
    from service.utils.retrieve_params import get_default_vsparams

    kwargs = get_default_vsparams(doc_path=doc_path)
    vsclient = get_vectorstore_from_type(**kwargs)

    _retriever = vsclient.get().as_retriever(
        search_type="mmr",
        search_kwargs={"k": k},
    )
    if not _retriever:
        return

    return MultiQueryRetriever.from_llm(
        retriever=_retriever, llm=get_llm(), include_original=False
    )
