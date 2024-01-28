from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    EmbeddingsFilter,
    LLMChainExtractor,
)
import logging
from core.llm import get_llm, get_embeddings
from core.query import get_retriever
import re, os

logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


def query(query: str, doc_path: str, k: int = 1):
    _retriever: ContextualCompressionRetriever = compression_retriever(
        doc_path, k, thre=0.5
    )
    _result = _retriever.get_relevant_documents(query)

    logging.info(f"retrieve output:\n{_result}")
    return list(map(lambda x: re.sub(r"\n+", "\n", x.page_content), _result))


def compression_retriever(doc_path: str, k: int = 1, thre: float = 0.5):
    kwargs = {
        "vd_name": "pinecone",
        "index_name": "manuals",
        "namespace": os.path.basename(doc_path),
        "doc_path": doc_path,
    }
    _retriever = get_retriever(search_type="mmr", k=k, **kwargs)
    if not _retriever:
        return

    _compressor = LLMChainExtractor.from_llm(
        llm=get_llm(), llm_chain_kwargs={"verbose": True}
    )
    # _compressor = EmbeddingsFilter(
    #     embeddings=get_embeddings(), similarity_threshold=thre
    # )
    return ContextualCompressionRetriever(
        base_compressor=_compressor, base_retriever=_retriever
    )
