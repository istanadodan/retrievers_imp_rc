from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain.tools.retriever import create_retriever_tool
from langchain.docstore.document import Document
from langchain.tools import tool
from langchain.agents.agent_types import AgentType
from langchain.utilities.sql_database import SQLDatabase
from core.llm import get_llm, get_embeddings
from core.db import get_vectorstore_from_type


def get_agent(agent_type="sql"):
    from service.utils.retrieve_params import get_default_vsparams

    llm = get_llm()

    # few shots 설정
    few_shots = {
        "List all members": "select * from users",
        "Who are over the age of 50": "select * from users where age > 50",
        "Who is the age of user input": "select age from users where age = ?",
    }
    db = SQLDatabase.from_uri(
        "postgresql+psycopg2://istana:istana@localhost:5432/llmdb"
    )
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    kwargs = get_default_vsparams(doc_path="", vd_name="faiss", namespace="member")
    vsclient = get_vectorstore_from_type(**kwargs)

    _docs = [
        Document(page_content=shot, metadata={"sql_query": few_shots[shot]})
        for shot in few_shots
    ]
    _retriever = vsclient.get().as_retriever(search_kwargs={"k": 1})
    _retriever.add_documents(_docs)

    # 검색기툴 설정
    tool = create_retriever_tool(
        retriever=_retriever,
        name="find_member",
        description="이 도구는 사용자가 회원정보를 조회하려는데 도움이 된다.",
    )

    if agent_type == "sql":
        return create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            extra_tools=[tool],
            suffix="회원정보를 요청할때 반드시 find_member함수를 호출하도록 할 것. 결과가 없으며,데이터베이스에서 관련성이 높은 스키마를 찾고 질의에 다시 답변할 것.",
        )

    # elif agent_type == "json":
    #     return create_json_agent(llm, vectorstore)
    # else:
    #     raise ValueError("Invalid agent type")


def query(query: str):
    agent = get_agent()
    return (query, agent.run(query))
