from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent


# -----------------------------------
# TOOL
# -----------------------------------

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


# -----------------------------------
# LLM
# -----------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# -----------------------------------
# AGENT
# -----------------------------------

agent = create_agent(
    model=llm,
    tools=[get_weather]
)


# -----------------------------------
# STATE / HISTORY
# -----------------------------------

state = {
    "messages": []
}


# -----------------------------------
# CHAT LOOP
# -----------------------------------

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break


    # Add the new user message to state
    state["messages"].append(
        {
            "role": "user",
            "content": question
        }
    )


    # Send the COMPLETE state to the agent
    result = agent.invoke(state)


    # Agent returns updated state
    state = result


    # Get the latest AI message
    ai_message = state["messages"][-1]

    print("\nAI:", ai_message.content)


# Then the agent/tool flow can become:

# Human: What is the weather in Hyderabad?
# AI: [calls get_weather("hyderabad")]
# Tool: 30°C and sunny
# AI: Hyderabad is 30°C and sunny.

# Human: How hot is it?
# AI: [calls get_weather("hyderabad")]
# Tool: 30°C and sunny
# AI: It is 30°C in Hyderabad.
