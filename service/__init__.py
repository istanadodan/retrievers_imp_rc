"""
    LLM으로 하여금 관련된 유사질의를 4개를 만들어 수행.
    질의 갯수는 default promptTemplate에 존재.
"""

from typing import List
from core.llm import get_llm
from langchain.prompts import PromptTemplate
from cmn.types.query import QueryType
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    RunnableConfig,
)
from service.callbacks.prompt_callabck import PromptStdOutCallbackHandler
from service.callbacks.reorder_callback import DocumentReorderCallbackHandler
from service.retrievers import (
    parent_document,
    contextual_compression,
    multi_query,
    ensembles,
)
from core.db import get_vectorstore_from_type
from langchain.memory import ConversationBufferWindowMemory

config = RunnableConfig(
    callbacks=[PromptStdOutCallbackHandler(), DocumentReorderCallbackHandler()]
)
chat_memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history")
chat_history = []
query_type_map = {
    QueryType.Multi_Query: multi_query.mquery_retriever,
    QueryType.Contextual_Compression: contextual_compression.compression_retriever,
    QueryType.Parent_Document: parent_document.pdoc_retriever,
    QueryType.Ensembles: ensembles.ensembles_retriever,
}


def simple_query(query: str):
    from langchain.chains.llm import LLMChain

    prompt_template = (
        "Tell me a simple answer to my question. \nQuestion: {query}\nAnswer:"
    )
    prompt = PromptTemplate(input_variables=["query"], template=prompt_template)
    simple_chain = LLMChain(
        llm=get_llm(), prompt=prompt, output_key="result", verbose=True
    )
    answer = simple_chain.invoke({"query": query})
    return query, {
        x: answer[x] for x in iter(answer) if x in ["result", "source_documents"]
    }


def webpage_summary(url: str, keyword: str, engine: QueryType, top_k: int):
    from service.loader import web_loader
    from pathlib import Path
    import re
    import datetime

    docs: List = web_loader.get_documents(url=url)
    save_path = Path(
        f"./public/url_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(save_path, encoding="utf-8", mode="w") as file:
        for doc in docs:
            if "." not in doc.page_content:
                continue
            _doc = re.sub(r"\n+", "\n", doc.page_content)
            file.write(_doc + "/n")

    return doc_summary(
        keyword,
        save_path,
        engine=engine,
        top_k=top_k,
    )


def doc_summary(keyword: str, path: str, engine: QueryType, top_k: int, **kwargs):
    from langchain.chains.summarize import load_summarize_chain
    from service.retrievers import multi_query, parent_document

    docs: List = []
    if engine == QueryType.Parent_Document:
        docs = parent_document.query(query=keyword, doc_path=path, k=top_k)
    else:
        docs = multi_query.query(query=keyword, doc_path=path, k=top_k)

    map_prompt_template = """
Please provide a concise summary of:
{text}
Requirements for Creating a Summary: 
{query}
Essential:
Rule 1. not generate any further content when there is no relevant information
Rule 2. Do not omit the main topic or entity that is the focus in the original passage.
Rule 3. summarize this text in 100 words with key details.
"""
    # 규칙 1. 반드시 주요 keyword가 포함된 문장으로 작성
    # 규칙 2. 주어를 빠뜨리지 않고 작성
    combine_prompt_template = """
Integrate these summaries into a coherent text: 
{text}
Write summaries by including repeated or important keywords in the presented context in Korean.
Requirements for Creating a Summary: 
{query}
Essential:
Rule 1. not generate any further content when there is no relevant information
Rule 2. Do not omit the main topic or entity that is the focus in the original passage.
Rule 3. summarize this text in over 300 words with key details.
Summary:
"""
    # keyword = None
    kwargs = {
        "llm": get_llm(),
        "chain_type": "map_reduce",
        "verbose": True,
        "memory": chat_memory,
        "input_key": "text",
        "output_key": "result",
        # "return_intermediate_steps": True,
        "token_max": 1000,
        "map_prompt": PromptTemplate(
            template=map_prompt_template,
            input_variables=["text"],
            partial_variables={"query": keyword},
        ),
        "combine_prompt": PromptTemplate(
            template=combine_prompt_template,
            input_variables=["text"],
            partial_variables={"query": keyword},
        ),
    }
    summary_chain = load_summarize_chain(**kwargs)
    summary_result = summary_chain.invoke({"text": docs}, return_only_outputs=True)
    return keyword, summary_result


def query(query: str, path: str, query_type: QueryType, k: int = 2):
    from langchain_core.output_parsers import StrOutputParser
    from service.tools import serp_api, mydata
    from langchain.agents import create_openai_functions_agent, AgentExecutor
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.document_transformers.long_context_reorder import LongContextReorder
    from langchain_core.messages import HumanMessage, AIMessage

    if not path:
        return

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
    _chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Answer the question based only on the following contexts and additional_info only if it is usefull:
{context}
Additional_Info: {addtional_info}
Write an answer in Korean.
Answer:
""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}"),
        ]
    )

    # LCEL을 사용하여 chain을 만들어본다.
    reordering = LongContextReorder()
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
            retrieved_docs=lambda x: _retriever.invoke(input=x["question"]),
        )
        | RunnablePassthrough().assign(
            context=lambda x: "context:\n"
            + "\ncontext:\n".join(
                [
                    doc.page_content
                    for doc in reordering.transform_documents(x["retrieved_docs"])
                ]
            ),
        )
        | RunnableParallel(
            result=_chat_prompt | get_llm() | StrOutputParser(),
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

    answer = _chain.invoke(
        {"question": query, "chat_history": chat_history[:1]},
        config=config,
    )

    chat_history.extend(
        [HumanMessage(content=query), AIMessage(content=answer["result"])]
    )

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
    return query, answer


def persist_vectorstore(path: str, vector_name: str = "chroma"):
    from service.utils.text_split import split_documents
    from service.loader.pdf_loader import get_documents
    import os
    from cmn.types.vectorstore import VectoreStoreInf

    # 문서 load
    docs = get_documents(path)
    # docs생성
    split_docs = split_documents(docs)

    kwargs = {
        "vd_name": vector_name,
        "index_name": "manuals",
        "namespace": os.path.basename(path),
        "docs": split_docs,
    }
    # vectorstore를 불러온다.
    _vstore: VectoreStoreInf = get_vectorstore_from_type(**kwargs)
    # save to db
    # _vstore.add(split_docs, **kwargs)
    # 저장한다.
    # _vstore.save(**kwargs)
