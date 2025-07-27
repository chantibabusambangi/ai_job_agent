from fastapi import FastAPI
from langchain_mcp_adapters.server import MCPServer
from resume_score_agent import youtube_utility

app = FastAPI()
server = MCPServer(app, func=youtube_utility, path="/mcp")
