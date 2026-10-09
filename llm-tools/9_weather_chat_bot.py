import json
import os
import re

from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, messages_from_dict, messages_to_dict
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
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
# 2. CHAT PROMPT TEMPLATE
# ==========================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are WeatherBot, a helpful weather assistant.

        Rules:
        1. Answer questions related to weather.
        2. Use the get_weather tool when the user asks
           about the current weather of a city.
        3. Never invent weather information.
        4. If the user asks an unrelated question,
           politely explain that you only handle weather-related questions.
        5. Use previous conversation history when it is relevant.
        6. Keep answers simple and conversational.
        """
    )
])

# Convert the system prompt template into a system prompt string.
system_prompt = prompt.format()


# ==========================================
# 3. LLM
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# ==========================================
# 4. CREATE AGENT
# ==========================================

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt=system_prompt
)


# ==========================================
# 5. PERSISTENCE CONFIGURATION
# ==========================================

HISTORY_DIR = Path("chat_history")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# In-memory store for active sessions.
store = {}


def get_session_history(session_id: str):

    # Prevent unsafe filenames.
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", session_id):
        raise ValueError("Invalid session ID")

    if session_id not in store:

        history_file = HISTORY_DIR / f"{session_id}.json"

        # Load existing conversation from JSON.
        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as file:
                saved_messages = json.load(file)

            messages = messages_from_dict(saved_messages)

            store[session_id] = InMemoryChatMessageHistory(
                messages=messages
            )

        else:
            store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


# ==========================================
# 6. SAVE HISTORY TO JSON
# ==========================================

def save_session_history(session_id: str):

    history = get_session_history(session_id)

    history_file = HISTORY_DIR / f"{session_id}.json"

    # Convert LangChain messages into JSON-serializable dictionaries.
    serialized_messages = messages_to_dict(history.messages)

    with open(history_file, "w", encoding="utf-8") as file:
        json.dump(
            serialized_messages,
            file,
            indent=4,
            ensure_ascii=False
        )


# ==========================================
# 7. WRAP AGENT WITH MESSAGE HISTORY
# ==========================================

agent_with_history = RunnableWithMessageHistory(
    agent,
    get_session_history,
    input_messages_key="messages",
    output_messages_key="messages",
)


# ==========================================
# 8. CHAT LOOP
# ==========================================

session_id = input("Enter your user/session ID: ").strip()

if not re.fullmatch(r"[a-zA-Z0-9_-]+", session_id):
    raise ValueError("Use only letters, numbers, underscores, and hyphens.")

print("\nWeatherBot is ready!")
print("Ask about the weather. Type 'exit' to quit.")

while True:

    question = input("\nYou: ").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    try:
        result = agent_with_history.invoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            },
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        )

        # Get the final AI response.
        final_message = result["messages"][-1]

        print("\nWeatherBot:", final_message.content)

        # Persist the complete conversation, including tool messages.
        save_session_history(session_id)

    except Exception as error:
        print(f"\nError: {error}")

print("\nChat ended.")
```
