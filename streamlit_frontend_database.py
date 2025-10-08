import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from LangGraph_Backened_Sqlite_Database import chatbot, retrive_all_threads, add_thread_to_db,rename_thread_in_db, model,model_with_tools
import uuid

# ************************************ Utility function **********************************
def generate_thread_id():
    return str(uuid.uuid4())

def generate_thread_name(user_message: str) -> str:
    """
    Use LLM to generate a short (max 5 words) title for this conversation,
    based only on the first user message
    """
    prompt = f"Generate a short, simple title(max 5 words) for this conversation: {user_message}"
    response = model.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def add_thread(thread_id, thread_name = "New Chat"):
    """Add new thread to session + DB"""
    if thread_id not in [t[0] for t in st.session_state["chat_threads"]]:
        st.session_state['chat_threads'].append((thread_id, thread_name))
        add_thread_to_db(thread_id, thread_name)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'], "New Chat")
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
add_thread(st.session_state['thread_id'], "New Chat")

# ************************************* Side Bar UI **************************************
st.sidebar.title("LangGraph ChatBot")
if st.sidebar.button("New Chat", key="new_chat_btn"):
    reset_chat()
st.sidebar.header("My Converations")

for thread_id, thread_name in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(thread_name,key=f"thread_btn_{thread_id}"):
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

# Rename feature
new_name = st.sidebar.text_input("Rename Current Chat", "", key="rename_input")
if st.sidebar.button("Save Name") and new_name:
    rename_thread_in_db(st.session_state['thread_id'], new_name)
    st.session_state['chat_threads'] = retrive_all_threads()


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

    # 🔹 Auto-generate a thread name if it's still "New Chat"
    current_thread_id = str(st.session_state['thread_id'])
    threads_dict = dict(st.session_state['chat_threads'])
    current_name = threads_dict.get(current_thread_id, "New Chat")

    if current_name == "New Chat":
        new_name = generate_thread_name(user_input)
        rename_thread_in_db(current_thread_id, new_name)
        st.session_state['chat_threads'] = retrive_all_threads()

    # Streaming
    config={"configurable":{"thread_id":st.session_state['thread_id']},"metadata":{"thread_id":st.session_state['thread_id']},"run_name":"chatting"}

    with st.chat_message(name="assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        AI_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append({"role":"assistant","content":AI_message})
