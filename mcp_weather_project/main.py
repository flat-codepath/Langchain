
import asyncio
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from mcp import Client, StdioServerParameters

load_dotenv()

async def main():

    # 1. Create the LLM
    model = init_chat_model(
        "google_genai:gemini-2.5-flash",
        temperature=0
    )

    # 2. Tell the MCP client how to launch the server
    server_params = StdioServerParameters(
        command="python",
        args=["weather_server.py"]
    )

    # 3. Connect to the MCP server
    async with Client(server_params) as client:

        # 4. Discover tools exposed by the server
        adapter = MCPAdapter(client)
        tools = await adapter.list_tools()

        print("Available tools:")
        for tool in tools:
            print(tool.name)

        # 5. Create the LangChain agent
        agent = create_agent(model, tools)

        # 6. Ask the agent a question
        result = await agent.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Hyderabad?"
                }
            ]
        })

        # 7. Print the final answer
        print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
