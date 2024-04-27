from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.agents import tool
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from models import get_llm, get_embeddings
from cmn.vectordb import get_vectorstore_from_type
from cmn.vectordb.retriever_default_param import get_default_vsparams
import logging
import os

os.environ["LANGCHAIN_PROJECT"] = "DEBATE AGENT"

# 토론자
# 주관자
"""
주관자는 토론자 목록을 가지고, 최초 토론개시를 한다.
토론자는 주관자에게 대화를 하고, 이 내용을 모든 토론자에게 전달한다.
토론자는 사용할 도구목록을 가진다.
"""

# __prompt = hub.pull("hwchase17/openai-functions-agent")


# 툴은 문서검색과 웹검색이 제공된다.
@tool
def doc_search1(query: str):
    """Refer to this document when you want to present a rebuttal to the proponents of medical school expansion."""

    kwargs = get_default_vsparams(
        doc_path="D:\\Projects_Python3\\Y2023\\lc_rag_v1\\assets\\download_docs\\no.txt"
    )
    _vstore = get_vectorstore_from_type(**kwargs).get()
    return _vstore.similarity_search_by_vector(get_embeddings().embed_query(query))


@tool
def doc_search2(query: str):
    """Refer to this document when you want to provide a rebuttal to the opposition to medical school expansion."""

    kwargs = get_default_vsparams(
        doc_path="D:\\Projects_Python3\\Y2023\\lc_rag_v1\\assets\\download_docs\\yes.txt"
    )
    _vstore = get_vectorstore_from_type(**kwargs).get()
    return _vstore.similarity_search_by_vector(get_embeddings().embed_query(query))


@tool
def web_search(query: str):
    """웹을 통해 주제를 검색한다."""
    pass


class Participant:
    topic: str
    id: int
    role: str
    message_history = []

    def __init__(
        self, mediator: "Mediator", id: int, name: str, role: str, tools: list[callable]
    ):
        self.mediator = mediator
        self.id = id
        self.name = name
        self.role = role
        self.initialize_agent(tools)

    def initialize_agent(self, tools: list[callable]):
        """ChatPromptTemplate(input_variables=['agent_scratchpad', 'input'],
        input_types={'chat_history': typing.List[typing.Union[langchain_core.messages.ai.AIMessage, langchain_core.messages.human.HumanMessage, langchain_core.messages.chat.ChatMessage, langchain_core.messages.system.SystemMessage, langchain_core.messages.function.FunctionMessage, langchain_core.messages.tool.ToolMessage]], 'agent_scratchpad': typing.List[typing.Union[langchain_core.messages.ai.AIMessage, langchain_core.messages.human.HumanMessage, langchain_core.messages.chat.ChatMessage, langchain_core.messages.system.SystemMessage, langchain_core.messages.function.FunctionMessage, langchain_core.messages.tool.ToolMessage]]}, messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant')), MessagesPlaceholder(variable_name='chat_history', optional=True),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')), MessagesPlaceholder(variable_name='agent_scratchpad')])
        """
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_openai_functions_agent(llm=get_llm(), tools=tools, prompt=prompt)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True
        )

    def read_topic(self, topic):
        """최초 주관자로부터 토의주제를 듣늗다"""
        # LLM에 질의를 확장토록 한다.
        self.topic = f"주재자:{topic}"
        self.message_history = []

    def listen(self, opponent, argument):
        """상대방 의견을 듣는다"""
        self.message_history.append(f"{opponent}:{argument}")

    def speech(self) -> None:
        """반론을 말한다"""
        claim = self.agent_executor.invoke(
            {"input": "\n".join([self.topic] + [self.name] + self.message_history)}
        )["output"]

        self.message_history.append(f"{self.name}:{claim}")
        # 의견을 개진. 주관자는 토론지속여부를 확인한다.
        self.mediator.pass_message(self.id, claim)
        # 발언권을 넘김.
        self.mediator.turn_speaker(self.id + 1)


class Mediator:
    participants: list[Participant]
    speaker: Participant

    def __init__(self, max_rebate_count: int = 5):
        self.participants = []
        self.max_rebate_count = max_rebate_count

    def add_participant(self, participant):
        self.participants.append(participant)

    def start_conversation(self, topic):
        import random

        starter = random.randint(0, len(self.participants) - 1)
        _topic = self._regenerate_topic(topic)

        for participant in self.participants:
            _topic = self._topic_format(participant, _topic)
            participant.read_topic(_topic)

        self.speaker = self.participants[starter]
        self.turn_speaker(self.speaker.id)

    def _topic_format(self, participant: Participant, topic: str):
        # 주제를 더 구체적으로 만들 수 있습니다.
        return f"""{topic}    
        Your name is {participant.name}.
        Your stance is as follows: {participant.role}
        Your goal is to persuade your conversation partner of your point of view.
        DO look up information with your tool to refute your partner's claims.
        DO cite your sources.
        DO NOT fabricate fake citations.
        DO NOT cite any source that you did not look up.
        DO NOT restate something that has already been said in the past.
        DO NOT add anything else.
        Stop speaking the moment you finish speaking from your perspective.
        If you or any of the other assistants have the final answer or don't have any answer,
        prefix your response with FINAL ANSWER to stop a argument."
        Answer in Korean.
        """

    def _regenerate_topic(self, topic):
        # 주제를 더 구체적으로 만들 수 있습니다.
        topic_specifier_prompt = [
            SystemMessage(content="You can make a topic more specific."),
            HumanMessage(
                content=f"""{topic}
            
            You are the moderator. 
            Please make the topic more specific.
            Please reply with the specified quest in 100 words or less.
            Speak directly to the participants: {*[p.name for p in self.participants],}.  
            Do not add anything else.
            Answer in Korean."""  # 다른 것은 추가하지 마세요.
            ),
        ]
        llm = get_llm()
        r = llm(topic_specifier_prompt)
        return r.content.strip()  # 양쪽 공백 제거.

    def turn_speaker(self, next_id):
        _next_speaker = self.decide_next_speaker(next_id)
        self.speaker = self.participants[_next_speaker]
        self.speaker.speech()

    def pass_message(self, id: int, message: str):
        if "FINAL ANSWER " in message:
            logging.info("토론을 종료합니다.\n{message}")
            # 토론을 종료한다.
            return

        for participant in self.participants:
            if participant.id == id:
                continue
            participant.listen(self.participants[id].name, message)

    def decide_next_speaker(self, id: int) -> int:
        """상대방의 반론을 듣고 다음 주관자를 결정한다."""
        return id % len(self.participants)


def run():
    mediator = Mediator(max_rebate_count=5)

    p1 = Participant(
        mediator=mediator,
        id=0,
        name="Doctor Union(의사협회)",
        role="""의사협회는 의료계의 권익을 보호하고 의사들의 이해관계를 대변하는 기관입니다. 의사들의 업무 환경과 안전을 중시하며, 환자 안전과 질 높은 의료 서비스를 제공하기 위해 노력합니다. 지금도 의사의 수는 충분하다는 입장이며, 의대 증원은 필수 의료나 지방 의료 활성화에 대한 실효성이 떨어집니다. 의대 증원을 감행할 경우, 의료 교육 현장의 인프라가 갑작스러운 증원을 감당하지 못할 것이란 우려를 표합니다.""",
        # tools=[doc_search1, web_search],
        tools=[doc_search1],
    )
    p2 = Participant(
        mediator=mediator,
        id=1,
        name="Government(대한민국정부)",
        role="""대한민국 정부는 국가의 행정을 책임지는 주체로서, 국민의 복지와 발전을 책임져야 합니다. 우리나라는 의사수가 절대 부족한 상황이며, 노인인구가 늘어나면서 의료 수요가 급증하고 있습니다. OECD 국가들도 최근 의사수를 늘렸습니다. 또한, 증원된 의사 인력이 필수의료와 지역 의료로 갈 수있도록 튼튼한 의료사고 안정망 구축 및 보상 체계의 공정성을 높이고자 합니다.""",
        # tools=[doc_search2, web_search],
        tools=[doc_search2],
    )
    mediator.add_participant(p1)
    mediator.add_participant(p2)

    mediator.start_conversation(
        "2024년 현재, 대한민국 대학교 의대 정원 확대 충원은 필요한가?"
    )


if __name__ == "__main__":
    run()
