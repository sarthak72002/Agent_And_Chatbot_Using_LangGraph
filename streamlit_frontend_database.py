import streamlit as st
from langchain_core.messages import HumanMessage
from LangGraph_Backened_Sqlite_Database import chatbot, retrive_all_threads
import uuid

# ************************************ Utility function **********************************
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state['chat_threads'].append(thread_id)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable":{"thread_id":thread_id}})
    return state.values.get("messages",[])
    

# ************************************* Session setup ************************************
if "message_history" not in st.session_state:
    st.session_state['message_history'] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state['chat_threads'] = retrive_all_threads()
add_thread(st.session_state['thread_id'])

# ************************************* Side Bar UI **************************************
st.sidebar.title("LangGraph ChatBot")
if st.sidebar.button("New Chat"):
    reset_chat()
st.sidebar.header("My Converations")

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id=thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            temp_messages.append({"role":role, "content":msg.content})
        st.session_state['message_history'] = temp_messages


# ************************************* Main UI ******************************************

# Loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message["role"]):
        st.text(message['content'])

user_input = st.chat_input(placeholder="Type here")

if user_input:
    # Add message to message history
    st.session_state['message_history'].append({"role":"user", "content":user_input})
    with st.chat_message(name="user"):
        st.text(body=user_input)

    # Streaming

    with st.chat_message(name="assistant"):
        AI_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream({"messages":[HumanMessage(content=user_input)]},config={"configurable":{"thread_id":st.session_state['thread_id']}},stream_mode="messages")
        )

    st.session_state["message_history"].append({"role":"assistant","content":AI_message})