from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from tavily import TavilyClient
load_dotenv()

tavily = TavilyClient()


@tool
def search(query: str):
    """Tool that searches over the internet."""
    return tavily.search(query=query)


llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

agent = create_agent(
    model=llm,
    tools=[search],
    system_prompt="You are a helpful assistant"
)


def main():
    result = agent.invoke(
        {"messages": [
            {"role": "user", "content": "What's the weather in Tokyo"}
        ]}
    )
    print(result['messages'][-1].text)


if __name__ == '__main__':
    main()
