from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import sqlite3
import requests

load_dotenv()
# Model
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b"
)
model = ChatHuggingFace(llm=llm)

# Tools
# Search Tool
search_tool = DuckDuckGoSearchRun(region = "us-en")

# Calculator Tool
@tool
def calculator(first_num:float, second_num:float, operation:str) -> dict:
    """
    Perform basic arthimatic operation on two numbers.
    Supported operations are add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error":"can't divide by zero"}
            result = first_num/second_num
        else:
            return {"error":f"Unsupported operation {operation}"}
        return {"first_num":first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error":str(e)}
    
# Stock Price Tool
@tool
def get_stock_price(symbol:str) -> dict:
    """
    Fetch the latest stock price of the given symbol (e.g. "AAPL", "TSLA")
    Using Alpha Vantage with the api key in URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=KFG2XM1BU5SCP2QL"
    r = requests.get(url=url)
    return r.json()

# Tool binding
tools = [get_stock_price, calculator, search_tool]
# Make the LLM tool aware
model_with_tools = model.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# chat node function
def chat_node(state: ChatState):
    """
    LLM node may answer or request a tool call
    """
    messages = state['messages']
    response = model_with_tools.invoke(messages)
    return {"messages":[response]}

# Tool node 
tool_node = ToolNode(tools=tools) # Execute the tools call


# Creating a sqlite database
conn = sqlite3.connect(database="Chatbot.db", check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

# Extra table for thread name
def init_db():
    c = conn.cursor()
    c.execute(
        """create table if not exists threads (
        thread_id text primary key,
        thread_name text)"""
    )
    conn.commit()

def add_thread_to_db(thread_id, thread_name = "New Chat"):
    c = conn.cursor()
    c.execute("insert or ignore into threads (thread_id, thread_name) values (?,?)",
              (str(thread_id), thread_name))
    conn.commit()

def rename_thread_in_db(thread_id, new_name):
    c = conn.cursor()
    c.execute("update threads set thread_name = ? where thread_id = ?", (new_name, thread_id))
    conn.commit()

def retrive_all_threads():
    """Return list of (thread_id, thread_name)"""
    c = conn.cursor()
    c.execute("select thread_id, thread_name from threads")
    rows = c.fetchall()
    return [(str(tid), name) for tid, name in rows]

# graph
graph = StateGraph(state_schema=ChatState)

# Add nodes
graph.add_node(node="chat_node", action=chat_node)
graph.add_node(node="tools", action=tool_node)

# Add edges
graph.add_edge(start_key=START, end_key="chat_node")
# Here if the LLM ask for a tool go to tool node else finish
graph.add_conditional_edges(source="chat_node", path=tools_condition)
graph.add_edge(start_key="tools",end_key="chat_node")

# Compile
chatbot = graph.compile(checkpointer=checkpointer)

# Intialize extra metadata table
init_db()
