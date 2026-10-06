from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Store histories for different users
store = {}


def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


chat = RunnableWithMessageHistory(
    llm,
    get_session_history
)


while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    response = chat.invoke(
        question,
        config={
            "configurable": {
                "session_id": "user_1"
            }
        }
    )

    print("AI:", response.content)