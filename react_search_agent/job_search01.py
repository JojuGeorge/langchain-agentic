
from typing import List
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()

# Structure data using Pydantic
# To represent the source of the answer


class Source(BaseModel):
    """Schema for source used by agent"""
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and source"""
    answer: str = Field(description="The agents answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer")


llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
agent = create_agent(
    model=llm,
    tools=[TavilySearch()],
    response_format=AgentResponse)


def main():
    result = agent.invoke(
        {"messages": [
            {"role": "user", "content": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
             }
        ]}
    )
    print(result, "\n\n")

    response = result["structured_response"]
    print(response.answer)
    print("\n\n")
    print(response.sources)


if __name__ == "__main__":
    main()
