"""
    패널티를 부여하여 방향을 제어.
    시간단위로 exp함수를 적용하고 있어서 하루만 지난 것은 decay_rate가 0.03에서도 검색되지 않는다.
"""

from typing import List
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores.faiss import FAISS
import logging


def query(query: str) -> str:
    from core.db import get_vectorstore_from_type

    _vs_wrapper = get_vectorstore_from_type(vd_name="faiss", store=InMemoryDocstore({}))

    """ decay_rate: 오래된 것에 대한 패널티 비율
        작을 수록 최신 날짜에 대한 가중치가 낮음.
        semantic_similarity + (1.0 - decay_rate) ^ hours_passed
    """
    retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=_vs_wrapper.get(), decay_rate=0.01, k=1
    )

    retriever.add_documents(_get_documents())

    _result = retriever.get_relevant_documents(query)
    logging.info(f"retrieve output:\n{_result}")

    return _result


def _get_documents():
    from datetime import timedelta, datetime

    yesterday = datetime.now() - timedelta(days=3)
    return [
        Document(
            page_content="영어는 훌룡합니다", metadata={"last_accessed_at": yesterday}
        ),
        Document(page_content="한국어는 훌룡합니다."),
    ]
