from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import create_agent


# ==========================================
# 1. TOOL
# ==========================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""

    weather_data = {
        "hyderabad": "30°C and sunny",
        "delhi": "32°C and clear",
        "mumbai": "29°C and cloudy",
    }

    return weather_data.get(
        city.lower(),
        f"Weather information is not available for {city}"
    )


# ==========================================
# 2. LLM
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# ==========================================
# 3. CREATE AGENT
# ==========================================

agent = create_agent(
    model=llm,
    tools=[get_weather]
)


# ==========================================
# 4. MEMORY STORE
# ==========================================

store = {}


def get_session_history(session_id: str):

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


# ==========================================
# 5. WRAP AGENT WITH MESSAGE HISTORY
# ==========================================

agent_with_history = RunnableWithMessageHistory(
    agent,
    get_session_history,
    input_messages_key="messages",
)


# ==========================================
# 6. CHAT LOOP
# ==========================================

session_id = "user_1"

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    result = agent_with_history.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        },
        config={
            "configurable": {
                "session_id": session_id
            }
        }
    )

    # Agent returns its state/messages.
    # The last message is the final AI response.
    final_message = result["messages"][-1]

    print("\nAI:", final_message.content)