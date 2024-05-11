"""
tcp      373      0 localhost:11434         localhost:39156         ESTABLISHED
tcp        0      0 localhost:39156         localhost:11434         ESTABLISHED
"""

# MODEL = "llama3-instruct-8b:latest"
# MODEL = "llama2:7b-chat"
MODEL = "llama3:latest"


def chat1(query: str):
    from langchain_community.llms.ollama import Ollama

    try:
        # default base_url is "http://localhost:11434"
        llm = Ollama(model=MODEL, verbose=True)
        # answer = llm.invoke("Why is the sky blue?")
        while True:
            query = input("query: ")
            if not query:
                print("bye!")
                break
            answer = llm.stream(query)
            print("Answer: ", end="")
            for token in answer:
                print(token, end="")
            else:
                print("")
    except Exception as e:
        print(e)
        print("ollama server is not running")
        exit(1)


def chat2(query):
    import ollama

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": query,
            },
        ],
    )
    print(response["message"]["content"])


if __name__ == "__main__":
    chat1("Why is the sky blue?")
