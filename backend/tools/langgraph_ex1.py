import json

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
)
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

from langchain_core.tools import tool
from typing import Annotated
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
import operator
from typing import Annotated, List, Sequence, Tuple, TypedDict, Union

from langchain.agents import create_openai_functions_agent
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
import functools

tavily_tool = TavilySearchResults(max_results=5)

repl = PythonREPL()


# Create Agents
def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    functions = [format_tool_to_openai_function(t) for t in tools]

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
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_functions(functions)


# Define tools
@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."]
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Succesfully executed:\n```python\n{code}\n```\nStdout: {result}"


# Create graph
# 1. Define State
# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, FunctionMessage):
        pass
    else:
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }


def tool_node(state):
    """This runs tools in the graph

    It takes in an agent action and calls that tool and returns the result."""
    tools = [tavily_tool, python_repl]
    tool_executor = ToolExecutor(tools)

    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    tool_input = json.loads(
        last_message.additional_kwargs["function_call"]["arguments"]
    )
    # We can pass single-arg inputs by value
    if len(tool_input) == 1 and "__arg1" in tool_input:
        tool_input = next(iter(tool_input.values()))
    tool_name = last_message.additional_kwargs["function_call"]["name"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input=tool_input,
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(
        content=f"{tool_name} response: {str(response)}", name=action.tool
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}


# Define Edge Logic
# Either agent can decide to end
def router(state):
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" in last_message.additional_kwargs:
        # The previus agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "end"
    return "continue"


def run():
    from core.llm import get_llm

    llm = get_llm()
    # llm = get_llm(model_name="gpt-4-1106-preview")

    # Define Tool Node
    # Research agent and node
    research_agent = create_agent(
        llm,
        [tavily_tool],
        system_message="You should provide accurate data for the chart generator to use.",
    )

    research_node = functools.partial(
        agent_node, agent=research_agent, name="Researcher"
    )

    # Chart Generator
    chart_agent = create_agent(
        llm,
        [python_repl],
        system_message="Any charts you display will be visible by the user.",
    )
    chart_node = functools.partial(
        agent_node, agent=chart_agent, name="Chart Generator"
    )

    # Define the Graph
    workflow = StateGraph(AgentState)

    workflow.add_node("Researcher", research_node)
    workflow.add_node("Chart Generator", chart_node)
    workflow.add_node("call_tool", tool_node)

    workflow.set_entry_point("Researcher")

    """1.노드, 2.함수(실행될 다음노드를 결정) 3.매핑"""
    workflow.add_conditional_edges(
        "Researcher",
        router,
        {"continue": "Chart Generator", "call_tool": "call_tool", "end": END},
    )
    workflow.add_conditional_edges(
        "Chart Generator",
        router,
        {"continue": "Researcher", "call_tool": "call_tool", "end": END},
    )

    workflow.add_conditional_edges(
        "call_tool",
        # Each agent node updates the 'sender' field
        # the tool calling node does not, meaning
        # this edge will route back to the original agent
        # who invoked the tool
        lambda x: x["sender"],
        {
            "Researcher": "Researcher",
            "Chart Generator": "Chart Generator",
        },
    )

    graph = workflow.compile()

    # Invoke
    for s in graph.stream(
        {
            "messages": [
                HumanMessage(
                    content="Fetch the UK's GDP over the past 5 years,"
                    " then draw a line graph of it."
                    " Once you code it up, finish."
                )
            ],
        },
        # Maximum number of steps to take in the graph
        {"recursion_limit": 5},
    ):
        print(s)
        print("----")


"""{'Researcher': {'messages': [HumanMessage(content='', additional_kwargs={'function_call': {'arguments': '{"query":"UK GDP from 2018 to 2023"}', 'name': 'tavily_search_results_json'}}, response_metadata={'token_usage': {'completion_tokens': 27, 'prompt_tokens': 221, 'total_tokens': 248}, 'model_name': 'gpt-4-1106-preview', 'system_fingerprint': 'fp_89f117abc5', 'finish_reason': 'function_call', 'logprobs': None}, name='Researcher', id='run-0fbb0c8a-3f4b-4736-a259-528223ec6a1d-0')], 'sender': 'Researcher'}}
----
{'call_tool': {'messages': [FunctionMessage(content='tavily_search_results_json response: [{\'url\': \'https://www.ons.gov.uk/economy/grossdomesticproductgdp\', \'content\': \'Quarter on Quarter growth: CVM SA %\\nChained Volume Measures (CVM)\\nGross Domestic Product: q-on-q4 growth rate CVM SA %\\nChained Volume 
Measures (CVM)\\nGross Domestic Product at market prices: Current price: Seasonally adjusted £m\\nCurrent Prices (CP)\\nGross Domestic Product: quarter on quarter growth rate: CP SA %\\nCurrent Prices (CP)\\nGross Domestic Product: q-on-q4 growth quarter growth: CP SA %\\nCurrent Prices (CP)\\nDatasets related to Gross Domestic Product (GDP)\\n A roundup of the latest data and trends on the economy, business and jobs\\nTime series related to Gross Domestic Product (GDP)\\nGross Domestic Product: chained volume measures: Seasonally adjusted £m\\nChained Volume Measures (CVM)\\nGross Domestic Product: Hide\\nData and analysis 
from Census 2021\\nGross Domestic Product (GDP)\\nGross domestic product (GDP) estimates as the main measure of UK economic growth based on the value of goods and services produced during a given period. Contains current and constant price data on the value of goods and services to indicate the economic performance of the UK.\\nEstimates of short-term indicators of investment in non-financial assets; business investment and asset and sector breakdowns of total gross fixed capital formation.\\n Monthly gross domestic product by gross value added\\nThe gross value added (GVA) tables showing the monthly and annual growths and indices as published within the monthly gross domestic product (GDP) statistical bulletin.\\n\'}, {\'url\': \'https://www.statista.com/statistics/281744/gdp-of-the-united-kingdom/\', \'content\': \'Industry Overview\\nDigital & Trend reports\\nOverview and forecasts on trending topics\\nIndustry & Market reports\\nIndustry and market insights and forecasts\\nCompanies & Products reports\\nKey figures and rankings about companies and products\\nConsumer & Brand reports\\nConsumer and brand insights and preferences in various industries\\nPolitics & Society reports\\nDetailed information about political and social topics\\nCountry & Region reports\\nAll key figures about countries and regions\\nMarket forecast and expert KPIs for 1000+ markets in 190+ countries & territories\\nInsights on consumer attitudes and behavior worldwide\\nBusiness information on 100m+ public and private companies\\nExplore Company Insights\\nDetailed information for 39,000+ online stores and marketplaces\\nDirectly accessible data for 170 industries from 150+ countries\\nand over 1\\xa0Mio. facts.\\n Transforming data into design:\\nStatista Content & Design\\nStrategy and business building for the data-driven economy:\\nGDP of the UK 1948-2022\\nUK economy expected to shrink in 2023\\nHow big is the UK economy compared to others?\\nGross domestic product of the United Kingdom from 1948 to 2022\\n(in million GBP)\\nAdditional Information\\nShow sources information\\nShow publisher information\\nUse Ask Statista Research Service\\nDecember 2023\\nUnited Kingdom\\n1948 to 2022\\n*GDP is displayed in real terms (seasonally adjusted chained volume measure with 2019 as the reference year)\\n Statistics on\\n"\\nEconomy of the UK\\n"\\nOther statistics that may interest you Economy of the UK\\nGross domestic product\\nLabor Market\\nInflation\\nGovernment finances\\nBusiness Enterprise\\nFurther 
related statistics\\nFurther Content: You might find this interesting as well\\nStatistics\\nTopics Other statistics on the topicThe UK economy\\nEconomy\\nRPI annual inflation rate UK 2000-2028\\nEconomy\\nCPI annual inflation rate UK 2000-2028\\nEconomy\\nAverage annual earnings for full-time employees in the UK 
1999-2023\\nEconomy\\nInflation rate in the UK 1989-2023\\nYou only have access to basic statistics.\\n Customized Research & Analysis projects:\\nGet quick analyses with our professional research service\\nThe best of the best: the portal for top lists & rankings:\\n\'}, {\'url\': \'https://www.ons.gov.uk/economy/grossdomesticproductgdp/timeseries/ihyp/pn2\', \'content\': \'Preliminary estimate of GDP time series (PGDP), released on 27 April 2018\\nPublications that use this data\\nContact details for this data\\nFooter links\\nHelp\\nAbout ONS\\nConnect with us\\nAll content is available under the Open Government Licence v3.0, except where otherwise stated Year on Year growth: CVM SA %\\nDownload full time series as:\\nDownload filtered time series as:\\nTable\\nNotes\\nFollowing a quality review it has been identified that the methodology used to estimate elements of purchased software within gross fixed capital formation (GFCF) has led to some double counting from 1997 onwards. GDP quarterly national accounts time series (QNA), released on 22 December 2023\\nIHYP: UK Economic Accounts 
time series (UKEA), released on 22 December 2023\\nIHYP: GDP first quarterly estimate time series\\n(PN2), released on 10 November 2023\\nIHYP: Year on Year growth: CVM SA %\\nSource dataset: GDP first quarterly estimate time series (PN2)\\nContact: Niamh McAuley\\nRelease date: 10 November 2023\\nView previous versions\\n %\\nFilters\\nCustom time period\\nChart\\nDownload this time seriesGross Domestic Product:\'}, {\'url\': \'https://www.macrotrends.net/global-metrics/countries/GBR/united-kingdom/gdp-gross-domestic-product\', \'content\': "U.K. gdp for 2021 was $3,122.48B, a 15.45% increase from 2020. U.K. gdp for 2020 was $2,704.61B, a 5.34% decline from 2019. U.K. gdp for 2019 was $2,857.06B, a 0.73% decline from 2018. GDP at purchaser\'s prices is the sum of gross value added by all resident producers in the economy plus any product taxes and minus any subsidies not included in ..."}, {\'url\': \'https://www.statista.com/topics/3795/gdp-of-the-uk/\', \'content\': \'Monthly growth of gross domestic product in the United Kingdom from January 2019 to November 2023\\nContribution to GDP 
growth in the UK 2023, by sector\\nContribution to gross domestic product growth in the United Kingdom in January 2023, by sector\\nGDP growth rate in the UK 
1999-2021, by country\\nAnnual growth rates of gross domestic product in the United Kingdom from 1999 to 2021, by country\\nGDP growth rate in the UK 2021, by region\\nAnnual growth rates of gross domestic product in the United Kingdom in 2021, by region\\nGDP growth of Scotland 2021, by local area\\nAnnual growth 
rates of gross domestic product in Scotland in 2021, by local (ITL 3) area\\nGDP growth of Wales 2021, by local area\\nAnnual growth rates of gross domestic product in Wales in 2021, by local (ITL 3) area\\nGDP growth of Northern Ireland 2021, by local area\\nAnnual growth rates of gross domestic product in Northern Ireland in 2021, by local (ITL 3) area\\nGDP per capita\\nGDP per capita\\nGDP per capita in the UK 1955-2022\\nGross domestic product per capita in the United Kingdom from 1955 to 2022 (in GBP)\\nAnnual GDP per capita growth in the UK 1956-2022\\nAnnual GDP per capita growth in the United Kingdom from 1956 to 2022\\nQuarterly GDP per capita in the UK 2019-2023\\nQuarterly GDP per capita in the United Kingdom from 1st quarter 2019 to 3rd quarter 2023 (in GBP)\\nQuarterly GDP per capita growth in the UK 2019-2023\\nQuarterly GDP per capita growth in the United Kingdom from 1st quarter 2019 to 3rd quarter 2023 (in GBP)\\nGDP per capita of the UK 1999-2021, by country\\nGross domestic product per capita of the United Kingdom from 1999 to 2021, by country (in GBP)\\nGDP per capita 
of the UK 2021, by region\\nGross domestic product per capita of the United Kingdom in 2021, by region (in GBP)\\nGlobal Comparisons\\nGlobal Comparisons\\nCountries with the largest gross domestic product (GDP) 2022\\n Monthly GDP of the UK 2019-2023\\nMonthly index of gross domestic product in the United Kingdom 
from January 2019 to November 2023 (2019=100)\\nGVA of the UK 2022, by sector\\nGross value added of the United Kingdom in 2022, by industry sector (in million GBP)\\nGDP of the UK 2021, by country\\nGross domestic product of the United Kingdom in 2021, by country (in million GBP)\\nGDP of the UK 2021, by region\\nGross domestic product of the United Kingdom in 2021, by region (in million GBP)\\nGDP of Scotland 2021, by local area\\nGross domestic product of Scotland in 2021, by local (ITL 3) area (in million GBP)\\nGDP of Wales 2021, by local area\\nGross domestic product of Wales in 2021, by local (ITL 3) area (in million 
GBP)\\nGDP of Northern Ireland 2021, by local area\\nGross domestic product of Northern Ireland in 2021, by local (ITL 3) area (in million GBP)\\nGDP growth\\nGDP growth\\nGDP growth forecast for the UK 2000-2028\\nForecasted annual growth of gross domestic product in the United Kingdom from 2000 to 2028\\nAnnual GDP growth in the UK 1949-2022\\nAnnual growth of gross domestic product in the United Kingdom from 1949 to 2022\\nQuarterly GDP growth of the UK 2019-2023\\nQuarterly growth of gross domestic product in the United Kingdom from 1st quarter 2019 to 3rd quarter 2023\\nMonthly GDP growth of the UK 2019-2023\\n Transforming data into design:\\nStatista Content & Design\\nStrategy and business building for the data-driven economy:\\nUK GDP - Statistics & Facts\\nUK economy expected to shrink in 2023\\nCharacteristics of UK GDP\\nKey insights\\nDetailed statistics\\nGDP of the UK 1948-2022\\nDetailed statistics\\nAnnual GDP growth 
in the UK 1949-2022\\nDetailed statistics\\nGDP per capita in the UK 1955-2022\\nEditor’s Picks\\nCurrent statistics on this topic\\nCurrent statistics on this topic\\nKey Economic Indicators\\nMonthly GDP growth of the UK 2019-2023\\nKey Economic Indicators\\nMonthly GDP of the UK 2019-2023\\nKey Economic Indicators\\nContribution to GDP growth in the UK 2023, by sector\\nRelated topics\\nRecommended\\nRecommended statistics\\nGDP\\nGDP\\nGDP of the UK 1948-2022\\nGross domestic product of the United Kingdom from 1948 to 2022 (in million GBP)\\nQuarterly GDP of the UK 2019-2023\\nQuarterly gross domestic product in the United Kingdom from 1st quarter 2019 to 3rd quarter 2023 (in million GBP)\\n The 20 countries with the largest gross domestic product (GDP) in 2022 (in billion U.S. dollars)\\nGDP of European countries in 2022\\nGross domestic product at current market prices of selected European countries in 2022 (in million euros)\\nReal GDP growth rates in Europe 2023\\nAnnual real gross domestic product (GDP) growth rate in European countries in 2023\\nGross domestic product (GDP) of Europe\\\'s largest economies 1980-2028\\nGross domestic product (GDP) at current prices of Europe\\\'s largest economies from 1980 to 2028 (in billion U.S dollars)\\nUnited Kingdom\\\'s share of global gross domestic product (GDP) 2028\\nUnited Kingdom (UK): Share of global gross domestic product (GDP) adjusted for 
Purchasing Power Parity (PPP) from 2018 to 2028\\nRelated topics\\nRecommended\\nReport on the topic\\nKey figures\\nThe most important key figures provide you with a compact summary of the topic of "UK GDP" and take you straight to the corresponding statistics.\\n Industry Overview\\nDigital & Trend reports\\nOverview and forecasts on trending topics\\nIndustry & Market reports\\nIndustry and market insights and forecasts\\nCompanies & Products reports\\nKey figures and rankings about companies and products\\nConsumer & Brand reports\\nConsumer and brand insights and preferences in various industries\\nPolitics & Society reports\\nDetailed information about political and social topics\\nCountry & Region reports\\nAll key figures about countries and regions\\nMarket forecast and 
expert KPIs for 1000+ markets in 190+ countries & territories\\nInsights on consumer attitudes and behavior worldwide\\nBusiness information on 100m+ public and private companies\\nExplore Company Insights\\nDetailed information for 39,000+ online stores and marketplaces\\nDirectly accessible data for 170 industries from 150+ countries\\nand over 1\\xa0Mio. facts.\\n\'}]', name='tavily_search_results_json')]}}"""
