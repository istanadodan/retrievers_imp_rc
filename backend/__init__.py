"""
    LLM으로 하여금 관련된 유사질의를 4개를 만들어 수행.
    질의 갯수는 default promptTemplate에 존재.
"""

from typing import List
from core.llm import get_llm
from langchain.prompts import PromptTemplate
from backend.cmn.types.query import QueryType
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    RunnableConfig,
)
from backend.callbacks.prompt_callabck import PromptStdOutCallbackHandler
from backend.callbacks.reorder_callback import DocumentReorderCallbackHandler
from backend.callbacks import ConsoleCallbackHandler
from backend.retrievers import (
    parent_document,
    contextual_compression,
    multi_query,
    ensembles,
    multi_vector,
)
from langchain.memory import ConversationBufferWindowMemory
from backend.cmn.types.vectorstore import VectoreStoreInf
from pathlib import Path

config = RunnableConfig(
    callbacks=[
        PromptStdOutCallbackHandler(),
        DocumentReorderCallbackHandler(),
        ConsoleCallbackHandler(),
    ]
)
chat_memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history")
chat_history = []
query_type_map = {
    QueryType.Multi_Query: multi_query.mquery_retriever,
    QueryType.Contextual_Compression: contextual_compression.compression_retriever,
    QueryType.Parent_Document: parent_document.pdoc_retriever,
    QueryType.Ensembles: ensembles.ensembles_retriever,
    QueryType.Multi_Vector: multi_vector.multivector_retriever3,
}

PUBLIC_PATH = "./assets/download_docs"


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
    from backend.cmn.loaders import get_documents_from_urls
    from backend.cmn.text_split import get_splitter
    import re
    import datetime

    docs: List = get_documents_from_urls(
        urls=[url], splitter=get_splitter(chunk_size=300)
    )
    save_path = Path(
        f"{PUBLIC_PATH}/url_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(save_path, encoding="utf-8", mode="w") as file:
        for doc in docs:
            _doc = re.sub(r"\n+", "\n", doc.page_content)
            file.write(_doc + "/n")

    return doc_summary(
        keyword,
        str(save_path.resolve()),
        engine=engine,
        top_k=top_k,
    )


def doc_summary(keyword: str, path: str, engine: QueryType, top_k: int, **kwargs):
    from langchain.chains.summarize import load_summarize_chain
    from backend.retrievers import multi_query, parent_document

    docs: List = []
    if engine == QueryType.Parent_Document:
        docs = parent_document.query(query=keyword, doc_path=path, k=top_k)
    else:
        docs = multi_query.query(query=keyword, doc_path=path, k=top_k)

    map_prompt_template = """
Please provide a concise summary of the following context within the <ctx> tag:
<ctx>
{text}
</ctx>
Points of Emphasis During Composition: 
{query}
Essential:
Rule 1. not generate any further content when there is no relevant information
Rule 2. Do not omit the main topic or entity that is the focus in the original passage.
Rule 3. summarize this text in 100 words with key details.
"""
    # 규칙 1. 반드시 주요 keyword가 포함된 문장으로 작성
    # 규칙 2. 주어를 빠뜨리지 않고 작성
    combine_prompt_template = """
Integrate these summaries within the <ctx> tag into a coherent text. 
If there is no context with the <ctx> tag, write only "관련정보를 찾을 수 없습니다.":
<ctx>
{text}
</ctx>
Points of Emphasis During Composition: 
{query}
Essential:
Rule 1. Not generate any further content when there is no relevant information
Rule 2. Do not omit the main topic or entity that is the focus in the original passage.
Rule 3. Summarize this text in over 300 words with key details.
Rol3 4. You must summarize in Korean. 
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
    return keyword, summary_result["result"]


def query(query: str, path: str, query_type: QueryType, k: int = 2):
    from langchain_core.output_parsers import StrOutputParser
    from backend.tools import serp_api, mydata
    from langchain.agents import create_openai_functions_agent, AgentExecutor
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.document_transformers.long_context_reorder import LongContextReorder
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

    if not path:
        return (None, None)

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
                """Answer the question based only on the following contexts or additional_info:
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
    # 처음과 마지막 부분 검색결과를 앞부분에 위치시킨다.
    reordering = LongContextReorder()
    
    _chain = (
        # RunnablePassthrough().assign(
        #     addtional_info=lambda x: agent_executor.invoke(
        #         {"input": x["question"]}
        #     )["output"],
        # )
        # | RunnablePassthrough().assign(
        #     retrieved_docs=lambda x: _retriever.invoke(input=x["question"]),
        # )
        # | RunnablePassthrough().assign(
        #     context=lambda x: "context:\n"
        #     + "\n\n".join(
        #         [
        #             doc.page_content
        #             for doc in reordering.transform_documents(x["retrieved_docs"])
        #         ]
        #     ),
        # )
        RunnableParallel(
            addtional_info=lambda x: agent_executor.invoke({"input": x["question"]})[
                "output"
            ],
            context=_retriever
            | RunnableLambda(
                lambda docs: "\n\n".join(
                    f"{doc.page_content}\nsource:\n{doc.metadata}"
                    for doc in reordering.transform_documents(docs)
                )
            ),
            question=lambda x: x["question"],
            chat_history=lambda x: x["chat_history"],
        )
        | RunnableParallel(
            result=_chat_prompt | get_llm() | StrOutputParser(),
            source_documents=RunnableLambda(lambda x: x["context"]),
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

    # chat_history.extend(
    #     BaseMessage(content=f"Question: {query}\nAnswer: {answer['result']}")
    # )
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
    return (query, answer["result"], answer["source_documents"])


def persist_to_vectorstore(
    path: str, is_pd_retriever: bool, vector_name: str = "chroma"
):
    from backend.cmn.text_split import split_documents
    from backend.cmn.loaders import get_documents_from_file
    from langchain.retrievers import ParentDocumentRetriever
    from backend.vectordb import get_vectorstore_from_type
    from backend.cmn.text_split import get_splitter
    import os

    # 문서 load
    docs = get_documents_from_file(path)
    # docs생성
    split_docs = split_documents(docs, chunk_size=300, chunk_overlap=0)

    kwargs = {
        "vd_name": vector_name,
        "index_name": "sharedocs",
        "namespace": os.path.basename(path),
    }
    # vectorstore를 불러온다.
    vsclient: VectoreStoreInf = get_vectorstore_from_type(**kwargs)

    # _vstore.add(split_docs, **kwargs)
    # 저장한다.
    # _vstore.save(**kwargs)

    if is_pd_retriever and len(split_docs) > 0:
        from langchain.storage import InMemoryStore
        from langchain_community.storage.astradb import AstraDBStore
        from langchain_astradb import AstraDBVectorStore

        COLLECTION_NAME = os.path.basename(path).split(".")[0]
        ASTRA_DB_ID = "ce493861-0a04-4778-a125-ab74f95f296e"
        ASTRA_DB_REGION = "us-east1"
        ASTRA_DB_KEYSPACE = "public"
        ASTRA_DB_APPLICATION_TOKEN = "AstraCS:qrjQhItsfzMsLyhxrPUfWCBv:4586d83f6186aef233fe078d0b5ddce7601b744fc126bfb36de421b96b7aa653"
        ASTRA_DB_API_ENDPOINT = (
            f"https://{ASTRA_DB_ID}-{ASTRA_DB_REGION}.apps.astra.datastax.com"
        )
        # ASTRA_DB_API_ENDPOINT = "https://ce493861-0a04-4778-a125-ab74f95f296e-us-east1.apps.astra.datastax.com"
        # {
        #   "clientId": "qrjQhItsfzMsLyhxrPUfWCBv",
        #   "secret": "C3NIXm2+U_7r3Zndth4NkjhvTqf7Ak9DaSht+eLbjde5m7wRS.uCvy3lAu_YQCOfZl+MxzCsXG-NY_g2QyO0IXih0lc1..7yJxE-R._BzO8CN9hZz3acvBvsixGedFsQ",
        #   "token": ""
        # }

        store = AstraDBStore(
            # embedding=get_embeddings(),
            collection_name=COLLECTION_NAME,
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=ASTRA_DB_KEYSPACE,
        )

        tmp_store = InMemoryStore()

        # vectorstore에 문서를 넣으면, 이 값이 parent doc이 되어 버린다.
        # child splitter를 사용한 문서를 vectorstore에 업데이트 한다.
        retriever = ParentDocumentRetriever(
            vectorstore=vsclient.get(),
            # parent_splitter=get_splitter(350),
            child_splitter=get_splitter(100),
            docstore=tmp_store,
        )
        # parent/child docstore 생성
        # ids=None 시, 오류
        retriever.add_documents(documents=split_docs)

        _store_data = [
            (k, {"page_content": v.page_content, "metadata": v.metadata})
            for k, v in tmp_store.store.items()
        ]

        store.mset(_store_data)

    else:
        # save to db
        vsclient.save(split_docs)
