from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool
from dotenv import load_dotenv
import aiosqlite
import asyncio
import threading
import truststore
truststore.inject_into_ssl()
load_dotenv()

# Dedicated async loop
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    return _submit_async(coro)


# ---------------- LLM ----------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)


# ---------------- MCP Tools ----------------

client = MultiServerMCPClient(
    {
        "arith": {
    "transport": "stdio",
    "command": "uv",
    "args": [
        "--directory",
        r"C:\Users\10857799\OneDrive - LTM\Desktop\local_mcp_server",
        "run",
        "python",
        "server.py"
    ]
}
        ,

        "expense": {
            "transport": "streamable_http",
            "url": "https://expense-tracker-by-krt.fastmcp.app/mcp"
        }
    }
)


def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception as e:
        import traceback
        traceback.print_exception(e)
        raise


mcp_tools = load_mcp_tools()
tools = mcp_tools

llm_with_tools = llm.bind_tools(tools) if tools else llm


# ---------------- State ----------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------- Nodes ----------------

async def chat_node(state: ChatState):
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None


# ---------------- Checkpointer ----------------

async def _init_checkpointer():
    conn = await aiosqlite.connect("chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())


# ---------------- Graph ----------------

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# ---------------- Helper ----------------

async def _alist_threads():
    all_threads = set()

    async for checkpoint in checkpointer.alist(None):
        all_threads.add(
            checkpoint.config["configurable"]["thread_id"]
        )

    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())
