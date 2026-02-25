import streamlit as st
from langgraphbackendsimple import workflow
from langchain_core.messages import HumanMessage

# Thread config for LangGraph checkpointing (persists conversation per thread_id)
CONFIG = {"configurable": {"thread_id": "1"}}

# Initialize message history in session state if not present
if 'message_history' not in st.session_state:
    st.session_state.message_history = []

# Render all previous messages (user + assistant) from history
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input at bottom of the app
user_input = st.chat_input("Enter your message:")

if user_input:
    # Append user message to history and display it
    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Stream LLM response token-by-token into the assistant chat bubble
    with st.chat_message("assistant"):
        ai_message = st.write_stream(  # Renders streamed chunks in the UI and returns final concatenated string
            message_chunk.content for message_chunk, metadata in workflow.stream(  # Generator: yield each message chunk's content
                {"messages": [HumanMessage(content=user_input)]},  # Input state: current user message
                config=CONFIG,  # Thread ID for checkpointing
                stream_mode="messages"  # Stream per-message updates (chunk by chunk)
            )
        )
    # Append full assistant response to history for reruns
    st.session_state.message_history.append({"role": "assistant", "content": ai_message})