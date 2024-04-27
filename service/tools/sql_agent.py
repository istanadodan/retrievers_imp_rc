from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain.tools.retriever import create_retriever_tool
from langchain.docstore.document import Document
from langchain.agents.agent_types import AgentType
from langchain.utilities.sql_database import SQLDatabase
from models import get_llm, get_embeddings
from cmn.vectordb import get_vectorstore_from_type


def query(query: str, k: int):
    agent_executor = get_agent(k)
    return (query, agent_executor.run(query))


def get_few_shots_retriever(top_k: int = 1):
    """
    이 도구는 사용자가 회원정보를 조회하려는데 도움이 된다.
    """
    from cmn.vectordb.retriever_default_param import get_default_vsparams

    # few shots 설정
    # key에 질의문내용을 넣고, value에 질의를 처리할 쿼리문 넣어준다.
    few_shots = {
        "List all members": "select * from users",
        "Who are over the age of 50": "select count(*) from users where age > 50",
        "대여순위별 목록": "select * from rental order by inventory_id desc",
    }

    _docs = [
        Document(page_content=shot, metadata={"sql_query": few_shots[shot]})
        for shot in few_shots
    ]

    kwargs = get_default_vsparams(doc_path="", vd_name="faiss", namespace="member")
    vsclient = get_vectorstore_from_type(**kwargs)

    _retriever = vsclient.get().as_retriever(search_kwargs={"k": top_k})

    _retriever.add_documents(_docs)
    return _retriever


def get_agent(k: int, agent_type="sql"):
    llm = get_llm()
    db = SQLDatabase.from_uri(
        "postgresql+psycopg2://istana:istana@localhost:5432/dvdrental"
    )
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # 검색기툴 설정
    retriever_tool = create_retriever_tool(
        retriever=get_few_shots_retriever(top_k=k),
        name="find_member",
        description="이 도구는 사용자가 회원정보를 조회할 때, 조회용 query를 제공한다.",
    )
    suffix_prompt = """
    대여정보를 요청할때 우선 find_member함수를 호출하여 쿼리문을 조회할. 조회결과가 없으며,데이터베이스에서 관련성이 높은 스키마를 찾아 질의에 다시 답변할 것.
    """
    if agent_type == "sql":
        return create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            extra_tools=[retriever_tool],
            suffix=suffix_prompt,
        )

    # elif agent_type == "json":
    #     return create_json_agent(llm, vectorstore)
    # else:
    #     raise ValueError("Invalid agent type")
