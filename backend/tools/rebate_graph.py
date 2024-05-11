import json
from langchain.agents import tool
from core.llm import get_llm, get_embeddings
from backend.vectordb import get_vectorstore_from_type
from backend.retrievers.retriever_default_param import get_default_vsparams

from langchain_core.messages import (
    FunctionMessage,
    HumanMessage,
    BaseMessage,
    SystemMessage,
)
from langchain.prompts import ChatPromptTemplate
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import END, StateGraph
import functools
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import MessagesPlaceholder

"""
주관자는 토론자 목록을 가지고, 최초 토론개시를 한다.
토론자는 주관자에게 대화를 하고, 이 내용을 모든 토론자에게 전달한다.
토론자는 사용할 도구목록을 가진다.
"""


# 툴은 문서검색과 웹검색이 제공된다.
@tool
def doc_search1(query: str):
    """Refer to this document when you want to present a rebuttal to the proponents of medical school expansion."""

    kwargs = get_default_vsparams(
        doc_path="D:\\Projects_Python3\\Y2023\\lc_rag_v1\\assets\\download_docs\\no.txt"
    )
    _vstore = get_vectorstore_from_type(**kwargs).get()
    return _vstore.similarity_search_by_vector(get_embeddings().embed_query(query), k=3)


@tool
def doc_search2(query: str):
    """Refer to this document when you want to provide a rebuttal to the opposition to medical school expansion."""

    kwargs = get_default_vsparams(
        doc_path="D:\\Projects_Python3\\Y2023\\lc_rag_v1\\assets\\download_docs\\yes.txt"
    )
    _vstore = get_vectorstore_from_type(**kwargs).get()
    return _vstore.similarity_search_by_vector(get_embeddings().embed_query(query), k=3)


@tool
def web_search(query: str):
    """웹을 통해 주제를 검색한다."""
    pass


# 노드를 생성한다
# 1.노드 생성
def agent_node(state, agent, name):
    result = agent.invoke(state)
    if isinstance(result, FunctionMessage):
        pass
    else:
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        "sender": name,
    }


def tool_node(state):
    """This runs tools in the graph"""
    tools = [doc_search1, doc_search2]
    tool_executor = ToolExecutor(tools)

    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]

    tool_input = json.loads(
        last_message.additional_kwargs["function_call"]["arguments"]
    )
    tool_name = last_message.additional_kwargs["function_call"]["name"]
    if len(tool_input) == 1 and "__args1" in tool_input:
        tool_input = next(iter(tool_input.values()))

    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
        tool=tool_name,
        tool_input=tool_input,
    )
    # We pass the ToolInvocation to the ToolExecutor
    response = tool_executor.invoke(action)
    # We construct a new message from the ToolExecutor result
    function_message = FunctionMessage(
        name=action.tool, content=f"{tool_name} response: {str(response)}"
    )
    return {"messages": [function_message]}


def router(state):
    """This is the router"""
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" in last_message.additional_kwargs:
        # The previus agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        return "end"
    # The previus agent is asking a question
    return "continue"


# 툴을 생성한다.

# 그래프를 개시한다
# 노드간 연결점을 만든다.
# 그래프를 컴파일한다
# 그래프를 실행한다.

AgentState = TypedDict(
    "AgentState",
    {"messages": Annotated[Sequence[BaseMessage], operator.add], "sender": str},
)


def create_agent(llm, tools: list, role: str):
    functions = [format_tool_to_openai_function(tool) for tool in tools]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{role}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(role=role)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_functions(functions)


def create_doctor_node(llm) -> callable:
    return functools.partial(
        agent_node,
        agent=create_agent(
            llm,
            tools=[doc_search1],
            role="의사협회는 의료계의 권익을 보호하고 의사들의 이해관계를 대변하는 기관입니다. 의사들의 업무 환경과 안전을 중시하며, 환자 안전과 질 높은 의료 서비스를 제공하기 위해 노력합니다. 지금도 의사의 수는 충분하다는 입장이며, 의대 증원은 필수 의료나 지방 의료 활성화에 대한 실효성이 떨어집니다. 의대 증원을 감행할 경우, 의료 교육 현장의 인프라가 갑작스러운 증원을 감당하지 못할 것이란 우려를 표합니다.",
        ),
        name="Doctor",
    )


def create_government_node(llm) -> callable:
    return functools.partial(
        agent_node,
        agent=create_agent(
            llm,
            tools=[doc_search2],
            role="대한민국 정부는 국가의 행정을 책임지는 주체로서, 국민의 복지와 발전을 책임져야 합니다. 우리나라는 의사수가 절대 부족한 상황이며, 노인인구가 늘어나면서 의료 수요가 급증하고 있습니다. OECD 국가들도 최근 의사수를 늘렸습니다. 또한, 증원된 의사 인력이 필수의료와 지역 의료로 갈 수있도록 튼튼한 의료사고 안정망 구축 및 보상 체계의 공정성을 높이고자 합니다.",
        ),
        name="Government",
    )


def run():
    llm = get_llm()

    workflow = StateGraph(AgentState)
    workflow.add_node("Doctor", create_doctor_node(llm=llm))
    workflow.add_node("Government", create_government_node(llm=llm))
    workflow.add_node("call_tool", tool_node)
    workflow.set_entry_point("Doctor")

    workflow.add_conditional_edges(
        "Doctor",
        router,
        {"continue": "Government", "call_tool": "call_tool", "end": END},
    )
    workflow.add_conditional_edges(
        "Government",
        router,
        {"continue": "Doctor", "call_tool": "call_tool", "end": END},
    )
    workflow.add_conditional_edges(
        "call_tool",
        lambda x: x["sender"],
        {"Docker": "Doctor", "Government": "Government"},
    )

    graph = workflow.compile()

    for s in graph.stream(
        {
            "messages": [
                HumanMessage(
                    content="2024년 현재, 대한민국 대학교 의대 정원 확대 충원은 필요한가?"
                )
            ],
        },
        {"recursion_limit": 5},
    ):
        print(s)
        print("-" * 15)


if __name__ == "__main__":
    run()
