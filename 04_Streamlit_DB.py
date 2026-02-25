# --- Frontend changes when using DB (vs in-memory) ---
# 1. Import: use langgraphbackenddatabase (workflow + retrieve_all_threads) instead of langgraphbackendsimple.
# 2. chat_threads init: set st.session_state['chat_threads'] = retrieve_all_threads() so sidebar loads
#    saved thread IDs from SQLite on first run (persists across app restarts).
# --- Bring in the tools we need ---
import re
import streamlit as st  # The library that draws our chat app on the web page
from langgraphbackenddatabase import workflow, retrieve_all_threads  # Workflow (LangGraph + SQLite checkpointer) and helper to load all thread IDs from DB
from langchain_core.messages import HumanMessage, AIMessage  # Types for "user said this" and "AI said this"
import uuid  # So we can create unique IDs for each chat (like giving each chat its own name tag)


def generate_thread_id():
    """Make a brand-new, one-of-a-kind ID for a new chat so we can tell chats apart."""
    thread_id = str(uuid.uuid4())  # uuid4() gives a random ID; str() turns it into text we can store
    return thread_id  # Send that ID back to whoever asked for it


def reset_chat():
    """Start a completely new chat: new ID, new list of threads, empty message history."""
    thread_id = generate_thread_id()  # Get a fresh ID for this new conversation
    st.session_state['thread_id'] = thread_id  # Remember this ID so the app knows "we're in this chat now"
    add_chat_thread(thread_id)  # Add this new chat to the sidebar list so we can click it later
    st.session_state.message_history = []  # Clear the screen: no old messages in this new chat


def add_chat_thread(thread_id):
    """Add a chat to our list of conversations if we don't already have it (no duplicates!)."""
    if thread_id not in st.session_state['chat_threads']:  # Only add if this ID isn't in the list yet
        st.session_state['chat_threads'].append(thread_id)  # Put the new thread_id at the end of the list


def load_conversation(thread_id):
    """Load all the messages we saved for a past chat so we can show them again when user clicks that chat."""
    state = workflow.get_state(config={"configurable": {"thread_id": thread_id}})  # Ask the workflow: "What do you have saved for this thread_id?"
    return state.values.get('messages', [])  # Give back the list of messages; if none, give an empty list []

def get_thread_display_title(thread_id, max_len=40):
    """Get first user message from a thread to show in sidebar instead of thread_id."""
    messages = load_conversation(thread_id)
    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = msg.content.strip()
            return (text[:max_len] + "...") if len(text) > max_len else text
    return thread_id[:8] + "..."  # fallback if no user message yet

# --- First time we run the app: create empty places to store our data ---
if 'message_history' not in st.session_state:  # Session state = our app's memory for this browser tab
    st.session_state.message_history = []  # message_history = list of {role, content} for what we show on screen

if 'thread_id' not in st.session_state:  # thread_id = which conversation we're in (like a folder for one chat)
    st.session_state['thread_id'] = generate_thread_id()  # Create one new ID and remember it

if 'chat_threads' not in st.session_state:  # Load list of conversation IDs from DB for the sidebar (persists across runs)
    st.session_state['chat_threads'] = retrieve_all_threads()  # Fetch all saved thread IDs from SQLite so sidebar shows past chats

add_chat_thread(st.session_state['thread_id'])  # Make sure the current chat's ID is in the sidebar list


# --- SIDEBAR: the strip on the left where we pick which chat to look at ---
st.sidebar.title('LangGraph Chatbot')  # Big title at the top of the sidebar
if st.sidebar.button('New Chat'):  # When user clicks "New Chat"...
    reset_chat()  # ...start a brand-new conversation (new ID, clear messages)
st.sidebar.header('My Conversations')  # Small header above the list of chats

# Show each saved conversation as a button; [::-1] means newest first (reverse the list)
for thread_id in st.session_state['chat_threads'][::-1]:
    display_title = get_thread_display_title(thread_id)  # User question (or truncated)
    if st.sidebar.button(display_title,key=thread_id):  # One button per chat; when user clicks one...
        st.session_state['thread_id'] = thread_id  # ...switch to that chat (this is now the "current" thread)
        messages = load_conversation(thread_id)  # Load from LangGraph all messages that were in that chat

        temp_messages = []  # We'll build a list in the format Streamlit needs: {role, content}

        for message in messages:  # For each message we loaded (user or AI)...
            if isinstance(message, HumanMessage):  # If it was something the user said...
                role = "user"  # ...mark it as "user"
            else:  # Otherwise it was the AI
                role = "assistant"  # ...mark it as "assistant"
            temp_messages.append({"role": role, "content": message.content})  # Add to our list with role + text

        st.session_state.message_history = temp_messages  # Replace what's on screen with this chat's messages


# --- MAIN AREA: show the current chat and let user type ---
for message in st.session_state.message_history:  # For every message in the current chat...
    with st.chat_message(message["role"]):  # Draw a bubble (user on one side, assistant on the other)
        st.write(message["content"])  # Put the message text inside the bubble

user_input = st.chat_input("Enter your message:")  # The text box at the bottom where user types

if user_input:  # Only do the rest when user actually typed something and hit Enter
    st.session_state.message_history.append({"role": "user", "content": user_input})  # Save user's message to history
    with st.chat_message("user"):  # Draw a user bubble
        st.write(user_input)  # Show what they just typed

    # Tell the workflow which conversation we're in so it saves replies to the right place
    CONFIG = {"configurable": {"thread_id": st.session_state['thread_id']}}

    with st.chat_message("assistant"):  # Draw an assistant bubble
        # Stream the AI reply word-by-word (like ChatGPT) and show it as it comes
        ai_message = st.write_stream(  # st.write_stream shows chunks as they arrive and returns the full text at the end
            message_chunk.content for message_chunk, metadata in workflow.stream(  # workflow.stream yields each piece of the reply
                {"messages": [HumanMessage(content=user_input)]},  # Send the user's message into the workflow
                config=CONFIG,  # Use our thread_id so the workflow knows which chat this belongs to
                stream_mode="messages"  # "Give me updates as messages (chunks) stream in"
            )
        )
    st.session_state.message_history.append({"role": "assistant", "content": ai_message})  # Save the full AI reply to history