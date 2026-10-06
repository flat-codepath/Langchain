from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.agents import create_agent


# ---------------------------------------
# TOOL
# ---------------------------------------

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


# ---------------------------------------
# LLM
# ---------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# ---------------------------------------
# AGENT
# ---------------------------------------

agent = create_agent(
    model=llm,
    tools=[get_weather]
)


# ---------------------------------------
# IN-MEMORY HISTORY
# ---------------------------------------

history = InMemoryChatMessageHistory()


# ---------------------------------------
# CHAT LOOP
# ---------------------------------------

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break


    # -----------------------------------
    # Add user's message to history
    # -----------------------------------

    history.add_user_message(question)


    # -----------------------------------
    # Give complete history to agent
    # -----------------------------------

    result = agent.invoke(
        {
            "messages": history.messages
        }
    )


    # -----------------------------------
    # Get the new AI messages
    # -----------------------------------

    messages = result["messages"]

    # The agent may have produced several
    # messages internally:
    #
    # HumanMessage
    # AIMessage (tool call)
    # ToolMessage
    # AIMessage (final answer)
    #
    # We only add new messages to history.


    # Find messages that are not already
    # stored in our history
    existing_count = len(history.messages)

    new_messages = messages[existing_count:]


    # Add the agent's new messages to history
    for message in new_messages:
        history.add_message(message)


    # -----------------------------------
    # Print final AI answer
    # -----------------------------------

    print("\nAI:", messages[-1].content)