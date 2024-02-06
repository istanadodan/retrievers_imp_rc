"""
    LLM으로 하여금 관련된 유사질의를 4개를 만들어 수행.
    질의 갯수는 default promptTemplate에 존재.
"""

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from core.llm import get_llm
from core.db import get_vectorstore_from_type
from service.retrievers import parent_document, contextual_compression, multi_query
from enum import Enum, auto
import logging
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser, SimpleJsonOutputParser
from service.tools import serp_api, mydata
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, SimpleJsonOutputParser
from langchain.schema.callbacks.stdout import StdOutCallbackHandler
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    RunnableConfig,
)
from service.callbacks.prompt_callabck import PromptStdOutCallbackHandler

config = RunnableConfig(callbacks=[PromptStdOutCallbackHandler()])


class QueryType(Enum):
    Multi_Query = auto()
    Contextual_Compression = auto()
    Parent_Document = auto()
    Simple_Query = auto()


def simple_query(query: str):
    prompt_template = (
        "Tell me a simple answer to my question. \nQuestion: {query}\nAnswer:"
    )
    prompt = PromptTemplate(input_variables=["query"], template=prompt_template)
    simple_chain = LLMChain(
        llm=get_llm(), prompt=prompt, output_key="result", verbose=True
    )
    answer = simple_chain.invoke({"query": query})
    return {x: answer[x] for x in iter(answer) if x in ["result", "source_documents"]}


def query(query: str, path: str, query_type: QueryType, k: int = 2):

    query_type_map = {
        QueryType.Multi_Query: multi_query.mquery_retriever,
        QueryType.Contextual_Compression: contextual_compression.compression_retriever,
        QueryType.Parent_Document: parent_document.pdoc_retriever,
    }

    _retriever = query_type_map[query_type](path, k=k)
    # 문서 포함 retriever
    if not _retriever:
        return {"result": "문서를 준비하지 못했습니다"}

    _prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "If the user query is not clear, you must respond firmly with 'Nothing'",
            ),
            ("user", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    tools = [mydata.get_mydata_company_info, serp_api.get_query_from_serpapi]
    agent = create_openai_functions_agent(tools=tools, llm=get_llm(), prompt=_prompt)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    # print(agent_executor.invoke({"input": query}))
    template = """Answer the question based only on the following context and additional_info only if it is usefull:
{context}
additional_info: {addtional_info}
Write in Korean.
Question: {question}
Answer:
"""
    # LCEL을 사용하여 chain을 만들어본다.
    _chain = (
        RunnablePassthrough().assign(
            addtional_info=lambda _query: agent_executor.invoke(
                {"input": _query["question"]}
            )["output"]
        )
        # {
        # "context": _retriever,
        # "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
        # }
        | RunnablePassthrough().assign(
            retrieved_docs=lambda x: _retriever.get_relevant_documents(x["question"]),
        )
        | RunnablePassthrough().assign(
            context=lambda x: "context:\n"
            + "\ncontext:\n".join([doc.page_content for doc in x["retrieved_docs"]]),
        )
        | RunnableParallel(
            result=PromptTemplate.from_template(template)
            | get_llm()
            | StrOutputParser(),
            source_documents=RunnableLambda(lambda x: x["retrieved_docs"]),
        )
        | RunnableLambda(
            lambda answer: {
                x: answer[x]
                for x in iter(answer)
                if x in ["result", "source_documents"]
            }
        )
    )

    answer = _chain.invoke({"question": query}, config=config)

    # 자동 쿼리생성하는 retriever
    # r_qa = RetrievalQA.from_llm(
    #     llm=get_llm(),
    #     retriever=_retriever,
    #     return_source_documents=True,
    #     verbose=True,
    #     input_key="question",
    #     llm_chain_kwargs={"verbose": True},
    # )
    # answer = r_qa.invoke({"question": query})
    # return {x: answer[x] for x in iter(answer) if x in ["result", "source_documents"]}
    return answer


def persist_vectorstore(path: str):
    from service.utils.text_split import split_documents
    from service.loader.pdf_loader import get_documents
    import os
    from cmn.types.vectorstore import VectoreStoreInf

    # 문서 load
    docs = get_documents(path)
    # docs생성
    split_docs = split_documents(docs)

    # vectorstore를 불러온다.
    _vstore: VectoreStoreInf = get_vectorstore_from_type(
        vd_name="pinecone", index_name="manuals", namespace=os.path.basename(path)
    )
    # save to db
    _vstore.add(split_docs)
    # 저장한다.
    _vstore.save()
