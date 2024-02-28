"""
    검색된 노드와 연관된 노드들을 찾는다.
    검색된 노드의  상위 노드에 딸린 자식노드가 연관성이 가장 높다고 가정한다.
    적정량의 child/parent node chunking크기를 고려한다. 예에서는 기본 (200/800)
"""

from typing import List
from langchain.retrievers import MultiVectorRetriever
from core.db import get_vectorstore_from_type
from langchain.storage import InMemoryStore
import logging


def query(query: str, doc_path: str, k: int = 3):
    import re

    # vectorstore에 문서를 넣으면, 이 값이 parent값이 되어 버린다.
    # vectorstore = get_vectorstore_from_type(vd_name="chroma")
    # sub_docs = vectorstore.similarity_search(query=query, k=1)

    retriever = multivector_retriever3(doc_path, k=k)
    # logging.info(f"store: {len(list(store.yield_keys()))}")

    _result = retriever.get_relevant_documents(query)
    logging.info(f"retrieve output:\n{_result}")

    return list(map(lambda x: re.sub(r"\n+", "\n", x.page_content), _result))


def multivector_retriever1(doc_path: str, k: int = 3):
    from service.utils.retrieve_params import get_default_vsparams
    import uuid
    from service.loaders import get_documents_from_file
    from service.utils.text_split import get_splitter
    from langchain.prompts import ChatPromptTemplate
    from core.llm import get_llm
    from langchain_core.output_parsers import StrOutputParser
    from langchain.docstore.document import Document

    docs = get_documents_from_file(doc_path, get_splitter(chunk_size=300))
    doc_ids = [str(uuid.uuid4()) for _ in docs]

    kwargs = get_default_vsparams(doc_path=doc_path, vd_name="faiss")
    vsclient = get_vectorstore_from_type(**kwargs)

    # 외부 DB로부터 doc를 읽어온다.
    # 1. original docs, 2. summary_docs
    store = InMemoryStore()
    id_key = "p_id"
    retriever = MultiVectorRetriever(
        vectorstore=vsclient.get(),
        docstore=store,
        search_type="mmr",
        search_kwargs={"k": k},
        kwargs={"verbose": True},
        id_key=id_key,
    )

    # parent document 역활
    sub_docs = []
    for i, doc in enumerate(docs):
        p_id = doc_ids[i]
        c_docs = get_splitter(chunk_size=100).split_documents([doc])
        for _doc in c_docs:
            _doc.metadata[id_key] = p_id
        sub_docs.extend(c_docs)

    # 나뉘어진 벡터를 추가한다.
    retriever.vectorstore.add_documents(sub_docs)

    # 검색은 자식문서를 대상으로 하고 조회결과는 그 부모문서를 대상으로 한다.
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    return retriever


# 요약문을 작성해서 질의대상으로 한다.
# 모든 문서에 대해 LLM을 사용해야하는 문제가 있다.
def multivector_retriever2(doc_path: str, k: int = 3):
    from service.utils.retrieve_params import get_default_vsparams
    import uuid
    from service.loaders import get_documents_from_file
    from service.utils.text_split import get_splitter
    from langchain.prompts import ChatPromptTemplate
    from core.llm import get_llm
    from langchain_core.output_parsers import StrOutputParser
    from langchain.docstore.document import Document

    docs = get_documents_from_file(doc_path, get_splitter(chunk_size=300))
    summary_chain = {
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
        | get_llm()
        | StrOutputParser()
    }
    summaries = summary_chain.batch(docs, {"max_concurrency": 5})
    doc_ids = [str(uuid.uuid4()) for _ in summaries]
    summary_docs = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]})
        for i, summary in enumerate(summaries)
    ]

    kwargs = get_default_vsparams(doc_path=doc_path, vd_name="faiss")
    vsclient = get_vectorstore_from_type(**kwargs)

    # 외부 DB로부터 doc를 읽어온다.
    # 1. original docs, 2. summary_docs
    store = InMemoryStore()
    id_key = "p_id"
    retriever = MultiVectorRetriever(
        vectorstore=vsclient.get(),
        docstore=store,
        search_type="mmr",
        search_kwargs={"k": k},
        kwargs={"verbose": True},
        id_key=id_key,
    )

    # 요약문을 벡터db에 추가한다.
    retriever.vectorstore.add_documents(summary_docs)

    # 검색은 요약문을 대상으로 하고 조회결과는 원문서를 대상으로 한다.
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    return retriever


# Hypothetical Queries
# 각 문서에 대해 질의가능한 내용으로 질의문을 작성해서 질의대상으로 한다.
def hypothetical_questions(arr: List[str]) -> str:
    return "\n\n".join(arr)


def multivector_retriever3(doc_path: str, k: int = 3):
    from service.utils.retrieve_params import get_default_vsparams
    import uuid
    from service.loaders import get_documents_from_file
    from service.utils.text_split import get_splitter
    from langchain.prompts import ChatPromptTemplate
    from core.llm import get_llm
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
    from langchain.docstore.document import Document

    docs = get_documents_from_file(doc_path, get_splitter(chunk_size=300))
    """
    funtion_call 함수의 schema를 설정한다.
     question은 LLM의 출력이면서 function_call 함수의 인자로 전달된다. 
     intermediate_step에서 이 인자를 인식하기 위한 이름이 "question"인 것이다.
    """
    functions = [
        {
            "name": "hypothetical_questions",
            "description": "Generate hypothetical questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["questions"],
            },
        }
    ]
    from langchain_core.runnables import RunnableLambda

    # function_call함수를 구현하여 사용한다.
    hy_questions_chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template(
            "Generate a list of exactly 3 hypothetical questions that the below document could be used to answer:\n\n{doc}"
        )
        | get_llm().bind(
            functions=functions, function_call={"name": "hypothetical_questions"}
        )
        | {
            "result": JsonKeyOutputFunctionsParser(key_name="questions"),
            "function_name": lambda x: x.additional_kwargs["function_call"]["name"],
        }
        | RunnableLambda(lambda x: eval(x["function_name"])(x["result"]))
        # |JsonKeyOutputFunctionsParser(key_name="questions")
        # | RunnableLambda(lambda x: "\n\n".join(x))
    )
    # function_call 함수없이 구현
    # hy_questions_chain = (
    #     {"doc": lambda x: x.page_content}
    #     | ChatPromptTemplate.from_template(
    #         "Generate a list of exactly 3 hypothetical questions that the below document could be used to answer:\n\n{doc}"
    #     )
    #     | get_llm()
    #     | StrOutputParser()
    # )

    """
    ["What was the author's first experience with programming like?",
    'Why did the author switch their focus from AI to Lisp during their graduate studies?',
    'What led the author to contemplate a career in art instead of computer science?']"""

    question_list = hy_questions_chain.batch(docs[:2], {"max_concurrency": 5})
    id_key = "p_id"

    doc_ids = [str(uuid.uuid4()) for _ in question_list]
    question_docs = [
        Document(page_content=question, metadata={id_key: doc_ids[i]})
        for i, question in enumerate(question_list)
    ]

    kwargs = get_default_vsparams(doc_path=doc_path, vd_name="faiss")
    vsclient = get_vectorstore_from_type(**kwargs)

    # 외부 DB로부터 doc를 읽어온다.
    # 1. original docs, 2. summary_docs
    store = InMemoryStore()
    retriever = MultiVectorRetriever(
        vectorstore=vsclient.get(),
        docstore=store,
        search_type="mmr",
        search_kwargs={"k": k},
        kwargs={"verbose": True},
        id_key=id_key,
    )

    # 생성한 질문내용을 벡터db에 추가한다.
    retriever.vectorstore.add_documents(question_docs)

    # 검색은 질문내용을 대상으로 하고 조회결과는 원문서를 대상으로 한다.
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    return retriever
