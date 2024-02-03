from dotenv import load_dotenv
from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI
import time
import os

load_dotenv()

chat_model_name = os.getenv("CHAT_MODEL_NAME")
client = OpenAI()


def create_assistant():
    """
    {'id': 'asst_9HAjl9y41ufsViNcThW1EXUS',
    'created_at': 1699828331,
    'description': None,
    'file_ids': [],
    'instructions': 'You are a personal math tutor. Answer questions briefly, in a sentence or less.',
    'metadata': {},
    'model': 'gpt-4-1106-preview',
    'name': 'Math Tutor',
    'object': 'assistant',
    'tools': []}
    """
    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
        model=chat_model_name,
    )
    return assistant.id


assistant_id = "asst_EPVEHCG74Y0uQKV6lPeZ7JGx" or create_assistant()


# Thread 생성, id 를 반환한다.
def create_thread() -> str:
    """
    'id': 'thread_bw42vPoQtYBMQE84WubNcJXG',
    'created_at': 1699828331,
    'metadata': {},
    'object': 'thread'
    """
    thread = client.beta.threads.create()
    return thread.id


thread_id = "thread_fFgc1wBQQk9mgziCsWQsToEQ" or create_thread()
# print(thread_id)


# 질의 메시지를 작성한다.
def create_message(
    thread_id: str,
    role: str,
    content: str,
    assistant_id: str = None,
) -> object:
    """
    'id': 'msg_IBiZDAWHhWPewxzN0EfTYNew',
    'assistant_id': None,
    'content': [{'text': {'annotations': [],
    'value': 'I need to solve the equation `3x + 11 = 14`. Can you help me?'},
    'type': 'text'}],
    'created_at': 1699828332,
    'file_ids': [],
    'metadata': {},
    'object': 'thread.message',
    'role': 'user',
    'run_id': None,
    'thread_id': 'thread_bw42vPoQtYBMQE84WubNcJXG'
    """
    return client.beta.threads.messages.create(
        # assistant_id=assistant_id,
        thread_id=thread_id,
        role=role,
        content=content,
    )


message = create_message(
    thread_id, "user", "I need to solve the equation `3x + 11 = 14`. Can you help me?"
)
print(message)


# 쓰레드에 질의를 보낸다.
# asynchronous operation
def create_run(assistant_id: str, thread_id: str) -> object:
    return client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )


# run object생성
run = create_run(assistant_id, thread_id)
run_id = run.id


def wait_on_run(thread_id: str, run: object) -> object:
    """
    'id': 'run_LA08RjouV3RemQ78UZXuyzv6',
    'assistant_id': 'asst_9HAjl9y41ufsViNcThW1EXUS',
    'cancelled_at': None,
    'completed_at': 1699828333,
    'created_at': 1699828332,
    'expires_at': None,
    'failed_at': None,
    'file_ids': [],
    'instructions': 'You are a personal math tutor. Answer questions briefly, in a sentence or less.',
    'last_error': None,
    'metadata': {},
    'model': 'gpt-4-1106-preview',
    'object': 'thread.run',
    'required_action': None,
    'started_at': 1699828332,
    'status': 'completed',
    'thread_id': 'thread_bw42vPoQtYBMQE84WubNcJXG',
    'tools': []
    """
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(run_id=run.id, thread_id=thread_id)
        time.sleep(0.5)
    return run


# 순환 조회
run = wait_on_run(thread_id, run)

print(run)

message = client.beta.threads.messages.list(thread_id=thread_id)
print(message)
"""
'data': [{'id': 'msg_S0ZtKIWjyWtbIW9JNUocPdUS',
   'assistant_id': 'asst_9HAjl9y41ufsViNcThW1EXUS',
   'content': [{'text': {'annotations': [],
      'value': 'Yes. Subtract 11 from both sides to get `3x = 3`, then divide by 3 to find `x = 1`.'},
     'type': 'text'}],
   'created_at': 1699828333,
   'file_ids': [],
   'metadata': {},
   'object': 'thread.message',
   'role': 'assistant',
   'run_id': 'run_LA08RjouV3RemQ78UZXuyzv6',
   'thread_id': 'thread_bw42vPoQtYBMQE84WubNcJXG'},
  {'id': 'msg_IBiZDAWHhWPewxzN0EfTYNew',
   'assistant_id': None,
   'content': [{'text': {'annotations': [],
      'value': 'I need to solve the equation `3x + 11 = 14`. Can you help me?'},
     'type': 'text'}],
   'created_at': 1699828332,
   'file_ids': [],
   'metadata': {},
   'object': 'thread.message',
   'role': 'user',
   'run_id': None,
   'thread_id': 'thread_bw42vPoQtYBMQE84WubNcJXG'}],
 'object': 'list',
 'first_id': 'msg_S0ZtKIWjyWtbIW9JNUocPdUS',
 'last_id': 'msg_IBiZDAWHhWPewxzN0EfTYNew',
 'has_more': False
"""

while True:
    query = input("질의내용을 입력해 주세요.\n")
    message = create_message(thread_id=thread_id, role="user", content=query)
    run = create_run(assistant_id=assistant_id, thread_id=thread_id)
    run = wait_on_run(thread_id, run)
    messages = client.beta.threads.messages.list(
        thread_id=thread_id, order="asc", after=message.id
    )
    print(messages.data[0].content[0].text.value)
