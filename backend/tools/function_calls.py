import datetime
import json
from dotenv import load_dotenv
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool

load_dotenv()


@tool
def get_flight_info(loc_origin, loc_destination):
    """Get flight information between two locations."""

    # Example output returned from an API or database
    flight_info = {
        "loc_origin": loc_origin,
        "loc_destination": loc_destination,
        "datetime": str(datetime.datetime.now() + datetime.timedelta(hours=2)),
        "airline": "KLM",
        "flight": "KL643",
    }

    return json.dumps(flight_info)


@tool
def send_back_email_for_complaint(email: str, subject: str, body: str) -> str:
    """Send email information and return the result."""

    # Example output returned from an API or database
    email_info = {
        "email": email,
        "subject": subject,
        "body": body,
    }
    # send_email(email_info)
    return "email has sent"


def query2(prompt: str):
    # from core.llm import get_llm
    from langchain.agents import (
        create_openai_functions_agent,
        AgentExecutor,
        initialize_agent,
    )

    # from langchain_community.chat_models import ChatOpenAI
    from langchain_openai import ChatOpenAI
    from langchain.prompts import (
        ChatPromptTemplate,
        MessagesPlaceholder,
    )

    # class Param(BaseModel):
    #     loc_origin: str = Field(description="The departure airport, e.g. DUS")
    #     loc_destination: str = Field(description="The destination airport, e.g. HAM")

    # messages = [
    #     SystemMessagePromptTemplate(
    #         prompt=PromptTemplate(
    #             input_variables=[], template="You are a helpful assistant"
    #         )
    #     ),
    #     MessagesPlaceholder(variable_name="chat_history", optional=True),
    #     HumanMessagePromptTemplate(
    #         prompt=PromptTemplate(input_variables=["input"], template="{input}")
    #     ),
    #     MessagesPlaceholder(variable_name="agent_scratchpad"),
    # ]

    # tools = [
    #     StructuredTool.from_function(
    #         func=get_flight_info,
    #         name="get_flight_info",
    #         description="get_flight_info(loc_origin, loc_destination) | Get flight information between two locations",
    #         args_schema=Param,
    #     ),
    # ]
    messages = [
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=1024)

    tools = [get_flight_info, send_back_email_for_complaint]

    prompt_ = ChatPromptTemplate.from_messages(messages)
    agent = create_openai_functions_agent(llm, tools, prompt_)
    callbacks = []
    # agent_executor = AgentExecutor(
    #     agent=agent,
    #     tools=tools,
    #     verbose=True,
    #     handle_parsing_errors=True,
    #     callbacks=callbacks,
    # )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        callbacks=callbacks,
    )

    # agent2 = initialize_agent(agent=agent, llm=llm)
    from langchain_core.messages import HumanMessage, AIMessage

    answer = agent_executor.invoke(
        {
            "input": prompt,
            "chat_history": [
                HumanMessage(content="hi! my name is bob"),
                AIMessage(content="Hello Bob! How can I assist you today?"),
            ],
            # "chat_history": [
            #     ("user", "hi! my name is bob"),
            #     ("assistant", "hello bob! how can i assist you today?"),
            # ],
        }
    )
    print(answer["output"])


def query(prompt: str):
    import openai

    function_descriptions = [
        {
            "name": "get_flight_info",
            "description": "Get flight information between two locations",
            "parameters": {
                "type": "object",
                "properties": {
                    "loc_origin": {
                        "type": "string",
                        "description": "The departure airport, e.g. DUS",
                    },
                    "loc_destination": {
                        "type": "string",
                        "description": "The destination airport, e.g. HAM",
                    },
                },
                "required": ["loc_origin", "loc_destination"],
            },
        }
    ]

    client = openai.Client()

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        # add function calling
        functions=function_descriptions,
        function_call="auto",  # specify the function call
    )
    output = completion.choices[0].message

    # origin = json.loads(output.function_call.arguments).get("loc_origin")
    # destination = json.loads(output.function_call.arguments).get("loc_destination")
    params = json.loads(output.function_call.arguments)

    chosen_function = eval(output.function_call.name)
    flight = chosen_function(**params)

    second_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
            {
                "role": "function",
                "name": output.function_call.name,
                "content": flight,
            },
        ],
        functions=function_descriptions,
    )

    # response = second_completion.choices[0].message.content
    print(second_completion)


def ask_and_reply(prompt):
    import openai

    """Give LLM a given prompt and get an answer."""

    function_descriptions = [
        {
            "name": "get_flight_info",
            "description": "Get flight information between two locations",
            "parameters": {
                "type": "object",
                "properties": {
                    "loc_origin": {
                        "type": "string",
                        "description": "The departure airport, e.g. DUS",
                    },
                    "loc_destination": {
                        "type": "string",
                        "description": "The destination airport, e.g. HAM",
                    },
                },
                "required": ["loc_origin", "loc_destination"],
            },
        }
    ]
    completion = openai.Client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        # add function calling
        functions=function_descriptions,
        function_call="auto",  # specify the function call
    )

    output = completion.choices[0].message
    return output


if __name__ == "__main__":
    user_prompt = """
This is Jane Harris. I am an unhappy customer that wants you to do several things.
First, I neeed to know when's the next flight from Amsterdam to New York.
Please proceed to book that flight for me.
Also, I want to file a complaint about my missed flight. It was an unpleasant surprise. 
Email me a copy of the complaint to jane@harris.com.
Please give me a confirmation after all of these are done.
"""
    query2(user_prompt)
