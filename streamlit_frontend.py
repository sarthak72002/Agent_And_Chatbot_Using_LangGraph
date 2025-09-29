import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph_backened import chatbot

if "message_history" not in st.session_state:
    st.session_state['message_history'] = []

for message in st.session_state['message_history']:
    with st.chat_message(message["role"]):
        st.text(message['content'])

user_input = st.chat_input(placeholder="Type here")

if user_input:
    # Add message to message history
    st.session_state['message_history'].append({"role":"user", "content":user_input})
    with st.chat_message(name="user"):
        st.text(body=user_input)
    
    thread_id = "1"
    response = chatbot.invoke({"messages":[HumanMessage(content=user_input)]},config={"configurable":{"thread_id":thread_id}})
    AI_message = response['messages'][-1].content
    st.session_state['message_history'].append({"role":"user", "content":AI_message})
    with st.chat_message(name="assistant"):
        st.text(body=AI_message)