from fastapi import FastAPI
from langchain_mcp_adapters.server import MCPServer
from email_agent import email_agent_node

app = FastAPI()
server = MCPServer(app, func=email_agent_node, path="/mcp")
