
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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


# 1. Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer clearly and simply."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])


# 2. Connect prompt and LLM using pipe operator
chain = prompt | llm


# 3. Add chat history management
chat = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)


# 4. Run the chat application
while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    response = chat.invoke(
        {"question": question},
        config={
            "configurable": {
                "session_id": "user_1"
            }
        }
    )

    print("AI:", response.content)