from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.callbacks import get_openai_callback

from core.llm import get_llm, get_embeddings
import streamlit as st
from langchain import hub


store = {}

model = get_llm()
embeddings = get_embeddings()
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 능숙한 어시스턴트입니다. 20자이내로 답변해주세요",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    print(f"store: {store}")
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


"""
SYSTEM
You are a helpful assistant

PLACEHOLDER
chat_history

HUMAN
{input}

PLACEHOLDER
agent_scratchpad
"""
agent_prompt = hub.pull("hwchase17/openai-tools-agent")


def run():
    if not st.session_state.file_path:
        return
    from backend.retrievers import multi_query
    from langchain.tools.retriever import create_retriever_tool
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    retriever = multi_query.mquery_retriever(doc_path=st.session_state.file_path, k=1)
    tool = create_retriever_tool(
        retriever=retriever,
        name="search_from_document",
        description="의대관련 조회 시에 반드시 참조할 것",
    )
    # agent_prompt.messages[0].prompt.template = (
    #     "당신은 유능한 어시스턴트입니다. 20자이내로 답변해주세요"
    # )
    agent = create_openai_tools_agent(llm=model, tools=[tool], prompt=agent_prompt)

    # runnable = (
    #     AgentExecutor.from_agent_and_tools(agent=agent, tools=[tool])
    #     | agent_prompt
    #     | model
    # )
    # from langchain.output_parsers import StructuredOutputParser

    with_message_history = RunnableWithMessageHistory(
        runnable=AgentExecutor.from_agent_and_tools(
            agent=agent, tools=[tool], verbose=True
        ),
        get_session_history=_get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    query = st.text_input("질문", key="query")

    if st.button("제출") and query:
        with get_openai_callback() as cb:
            st.session_state.conversation.append(
                dict(
                    user=query,
                    ai=with_message_history.invoke(
                        {"input": query},
                        config={"configurable": {"session_id": "st123"}},
                    )["output"],
                    source="",
                )
            )
            st.session_state.token_usage = cb.__dict__

        st.rerun()
