"""
    정제가 되어있는 자료에 대해 유용. 일반 PDF등 문서에는 효용적이지 않음.
"""
from typing import List
from langchain.docstore.document import Document
from langchain.retrievers import SelfQueryRetriever
import logging


def query(query: str):
    from core.db import get_vectorstore
    from core.llm import get_llm

    docs: List[Document] = get_documents()
    vectorstore = get_vectorstore(docs)

    search_schema = get_fields_info()
    document_content_description = "Brief summary of a movie"
    retriever = SelfQueryRetriever.from_llm(
        llm=get_llm(),
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=search_schema,
        verbose=True,
        # search_kwargs={"k": 4}, #최대 조회건 수
    )

    _result = retriever.get_relevant_documents(query)
    logging.info(f"retrieve output:\n{_result}")
    return list(map(lambda x: x.page_content, _result))


def get_fields_info():
    from langchain.chains.query_constructor.schema import AttributeInfo

    _schemas = [
        AttributeInfo(
            name="genre",
            description="The genre of the movie, One of ['science fiction','comedy','drama]",
            type="string",
        ),
        AttributeInfo(
            name="year",
            description="The year the movie was released",
            type="integer",
        ),
        AttributeInfo(
            name="director",
            description="The name of the movie director",
            type="string",
        ),
        AttributeInfo(
            name="rating",
            description="The review score for the movie",
            type="float",
        ),
    ]
    return _schemas


def get_documents():
    docs = [
        Document(
            page_content="A bunch of scientists bring back dinosaurs and mayhem breaks llose",
            metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
        ),
        Document(
            page_content="Leo DiCaprio gets lost in a dream ...",
            metadata={"year": 2010, "rating": 8.2, "genre": "romangse"},
        ),
        Document(
            page_content="A psychologist / detective gets lost in a series of dreams",
            metadata={
                "year": 2006,
                "rating": 8.6,
                "genre": "science fiction",
                "director": "Christopher Nolan",
            },
        ),
        Document(
            page_content="A bunch of normal-sized woman are supremely wholesome and some men pine after them",
            metadata={
                "year": 2019,
                "rating": 8.3,
                "genre": "science fiction",
                "director": "Greta Gerwig",
            },
        ),
    ]
    return docs
