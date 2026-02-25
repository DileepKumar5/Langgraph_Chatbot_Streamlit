import streamlit as st
from langgraphbackendsimple import workflow
from langchain_core.messages import HumanMessage


CONFIG = {"configurable": {"thread_id": "1"}}

if 'message_history' not in st.session_state:
    st.session_state.message_history = []
    
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Enter your message:")

if user_input:
    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
        
    response = workflow.invoke({"messages": [HumanMessage(content=user_input)]}, config=CONFIG)
    ai_message = response["messages"][-1].content
    st.session_state.message_history.append({"role": "assistant", "content": ai_message})
    with st.chat_message("assistant"):
        st.write(ai_message)