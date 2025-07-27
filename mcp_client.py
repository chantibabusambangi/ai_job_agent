import os
import asyncio
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

async def build_agent():
    client = MultiServerMCPClient({
        "resume":  { "url": "http://localhost:8101/mcp", "transport":"streamable_http" },
        "youtube": { "url": "http://localhost:8102/mcp", "transport":"streamable_http" },
        "email":   { "url": "http://localhost:8103/mcp", "transport":"streamable_http" },
    })
    tools = await client.get_tools()
    llm   = ChatGroq(model="qwen-qwq-32b", api_key=os.getenv("GROQ_API_KEY"))
    agent = create_react_agent(llm, tools)
    return agent

# Utility to be used in Streamlit
def get_agent():
    return asyncio.get_event_loop().run_until_complete(build_agent())
