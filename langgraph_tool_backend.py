# --- Backend changes when using DB (vs in-memory) ---
# 1. Use SqliteSaver with a sqlite3 connection so checkpoints (conversations) persist to chat_bot.db.
# 2. Pass checkpointer into graph.compile(checkpointer=checkpointer) so the workflow reads/writes SQLite.
# 3. Add retrieve_all_threads() that lists thread_ids from checkpointer.list() so the frontend can load
#    the sidebar with saved chats. Frontend must call this on init and use workflow + retrieve_all_threads.
# ---
import sqlite3
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")

# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator (first_number: int, second_number: int, operation: str) -> dict:
    """
    Perform a basic calculation with two numbers.
    supported operations: add, subtract, multiply, divide
    """
    try:
        if operation == "add":
            result = first_number + second_number
        elif operation == "subtract":
            result = first_number - second_number
        elif operation == "multiply":
            result = first_number * second_number
        elif operation == "divide":
            if second_number == 0:
                return {"error": "Cannot divide by zero"}
            result = first_number / second_number
        else:
            return {"error": "Invalid operation"}

    
        return{"first_number": first_number, "second_number": second_number, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
    
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g., 'AAPL', 'TSLA', 'GOOG')
    using Alpha Vantage API with API key in the url
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=0AKGI8OUVVTCGWEM"
    r = requests.get(url)
    return r.json()
    
tools = [calculator, get_stock_price, search_tool]

llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
def chat_node(state: ChatState):
    """ LLM node that may answer or request a tool call"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tools_node = ToolNode(tools)

conn = sqlite3.connect("chat_bot.db", check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tools_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
workflow = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
        
