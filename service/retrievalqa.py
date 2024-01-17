from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.retrievers import MultiQueryRetriever
from core.llm import get_llm
from core.query import get_retriever
import logging

logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


def query(query: str):
    # 문서 포함 retriever
    _retriever = get_retriever(k=1, path="./public/R2304581.pdf")

    # 자동 쿼리생성하는 retriever
    mul_retriever: MultiQueryRetriever = MultiQueryRetriever.from_llm(
        retriever=_retriever, llm=get_llm(), include_original=False
    )

    r_qa = RetrievalQA.from_llm(
        llm=get_llm(),
        retriever=mul_retriever,
        return_source_documents=True,
        verbose=True,
    )
    answer = r_qa.invoke({"query": query})
    return {x: answer[x] for x in iter(answer) if x in ["result", "source_documents"]}
