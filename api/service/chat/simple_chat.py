from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from api.cmn.core.llms import get_llm

def chat(question:str):
    llm = get_llm(type='local')

    prompt = PromptTemplate.from_template(
        """
        <s>Below is an instruction that describes a task. Write a response that appropriately completes the request in Korean.</s>
        <s>Question:</s>
        {question}

        <s>Answer:</s>"""
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"question":question})
