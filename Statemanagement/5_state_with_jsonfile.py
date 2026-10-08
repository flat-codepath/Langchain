import os
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import messages_to_dict, messages_from_dict
from langchain_core.runnables.history import RunnableWithMessageHistory


# -----------------------------
# LLM
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# -----------------------------
# History folder
# -----------------------------

HISTORY_DIR = "chat_history"

os.makedirs(HISTORY_DIR, exist_ok=True)


# -----------------------------
# In-memory store
# -----------------------------

store = {}


# -----------------------------
# Load history from JSON
# -----------------------------

def get_session_history(session_id):

    # Already loaded in memory
    if session_id in store:
        return store[session_id]

    file_path = os.path.join(
        HISTORY_DIR,
        f"{session_id}.json"
    )

    history = InMemoryChatMessageHistory()

    # If user's file exists
    if os.path.exists(file_path):

        with open(file_path, "r", encoding="utf-8") as file:

            data = json.load(file)

        # Convert dictionary -> LangChain messages
        messages = messages_from_dict(data)

        history.add_messages(messages)

    # Store in memory
    store[session_id] = history

    return history


# -----------------------------
# Save history to JSON
# -----------------------------

def save_session_history(session_id):

    history = store[session_id]

    file_path = os.path.join(
        HISTORY_DIR,
        f"{session_id}.json"
    )

    # Convert LangChain messages -> dictionary
    data = messages_to_dict(history.messages)

    with open(file_path, "w", encoding="utf-8") as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


# -----------------------------
# RunnableWithMessageHistory
# -----------------------------

chat = RunnableWithMessageHistory(
    llm,
    get_session_history
)


# -----------------------------
# Chat
# -----------------------------

session_id = input("Enter user ID: ")


while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    response = chat.invoke(
        question,
        config={
            "configurable": {
                "session_id": session_id
            }
        }
    )

    print("AI:", response.content)

    # Save conversation
    save_session_history(session_id)