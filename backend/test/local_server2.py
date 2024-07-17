from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf",
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

prompt = PromptTemplate.from_template(
    """
    <s>Below is an instruction that describes a task. Write a response that appropriately completes the request in Korean.</s>
    <s>Question:</s>
    {question}

    <s>Answer:</s>"""
)

chain = prompt | llm | StrOutputParser()

prompt = '기계식 키보드를 갖고싶어요'
chain.invoke({"question":prompt})
