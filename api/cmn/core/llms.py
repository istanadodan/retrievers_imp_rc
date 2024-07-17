from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from api.cmn.settings import settings

def get_llm(type:str) -> BaseChatModel:
    if type == "local":            
        return ChatOpenAI(
            base_url=settings.LOCAL_BASE_URL,
            api_key=settings.LOCAL_API_KEY,
            model=settings.LOCAL_LLM_MODEL,
            streaming=True,
            verbose=True,
            temperature=0.7,
            max_tokens=512,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0,
            # stop=["\n\n"],
            callbacks=[StreamingStdOutCallbackHandler()],
        )
    else:
        raise NotImplementedError(f"Unknown model type: {type}")
