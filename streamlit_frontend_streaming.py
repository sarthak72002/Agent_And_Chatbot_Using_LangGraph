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

    # Streaming
    thread_id = "1"

    with st.chat_message(name="assistant"):
        AI_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream({"messages":[HumanMessage(content=user_input)]},config={"configurable":{"thread_id":thread_id}},stream_mode="messages")
        )

    st.session_state["message_history"].append({"role":"assistant","content":AI_message})
   