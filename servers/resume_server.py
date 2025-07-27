from fastapi import FastAPI, Request
from langchain_mcp_adapters.server import MCPServer
from resume_score_agent import score_resume_vs_jd  # your existing logic

app = FastAPI()
server = MCPServer(app, func=score_resume_vs_jd, path="/mcp")
