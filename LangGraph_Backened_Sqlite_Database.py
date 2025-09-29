from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b"
)
model = ChatHuggingFace(llm=llm)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):

    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages":[response]}


# Creating a sqlite database
conn = sqlite3.connect(database="Chatbot.db", check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

# graph
graph = StateGraph(state_schema=ChatState)

# Add nodes
graph.add_node(node="chat_node", action=chat_node)

# Add edges
graph.add_edge(start_key=START, end_key="chat_node")
graph.add_edge(start_key="chat_node", end_key=END)

# compile
chatbot = graph.compile(checkpointer=checkpointer)


def retrive_all_threads():
    all_threads = set()
    for i in checkpointer.list(None):
        all_threads.add(i.config['configurable']['thread_id'])
    return list(all_threads)