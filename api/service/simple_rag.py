from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from api.cmn.core.llms import get_llm, get_embedding
from api.cmn.prepocess.preprocess import RagFileType
from api.cmn.logger import Logger
from pathlib import Path

logger = Logger.getLogger(__name__)

def rag(file_path: Path):
    embed_model = get_embedding(type="local")

    from api.cmn.prepocess.preprocess import DefaultPreProcessor
    from api.cmn.vectordb.load import VectorDbLoader

    preprocessor = DefaultPreProcessor(
        embed_model=embed_model, chunk_params={"chunk_size": 500, "chunk_overlap": 0}
    )

    docs = preprocessor.process(file_type=RagFileType.PDF, file_path=file_path)
    
    # persistence
    vdb = VectorDbLoader(name='chroma')
    # chain = prompt | llm | StrOutputParser()

    return docs
