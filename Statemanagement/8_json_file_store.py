
import json
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.runnables.history import RunnableWithMessageHistory


# Folder where chat histories will be saved
HISTORY_DIR = Path("chat_histories")
HISTORY_DIR.mkdir(exist_ok=True)


# Custom JSON-based chat history
class JSONChatMessageHistory(BaseChatMessageHistory):

    def __init__(self, session_id):
        self.file_path = HISTORY_DIR / f"{session_id}.json"

    @property
    def messages(self):
        if not self.file_path.exists():
            return []

        with open(self.file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return messages_from_dict(data)

    def add_messages(self, messages):
        # Read existing history
        history = self.messages

        # Add new Human/AI messages
        history.extend(messages)

        # Convert messages into JSON-compatible dictionaries
        data = messages_to_dict(history)

        # Save to hard disk
        with open(self.file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)

    def clear(self):
        with open(self.file_path, "w", encoding="utf-8") as file:
            json.dump([], file)


# Create LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# LangChain calls this function with the session_id
def get_session_history(session_id):
    return JSONChatMessageHistory(session_id)


chat = RunnableWithMessageHistory(
    llm,
    get_session_history
)


# Example: use a consistent session ID to resume the same chat
session_id = input("Enter conversation ID: ").strip()

if not session_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in session_id):
    raise ValueError("Use only letters, numbers, underscores, and hyphens.")

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
