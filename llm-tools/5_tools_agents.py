from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent


# --------------------------------
# TOOL 1
# --------------------------------
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


# --------------------------------
# TOOL 2
# --------------------------------
@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""

    try:
        result = eval(expression)
        return str(result)

    except Exception:
        return "Invalid mathematical expression"


# --------------------------------
# TOOL 3
# --------------------------------
@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""

    time_data = {
        "hyderabad": "10:30 AM",
        "delhi": "10:35 AM",
        "mumbai": "10:32 AM",
    }

    return time_data.get(
        city.lower(),
        f"Time information is not available for {city}"
    )


# --------------------------------
# CREATE LLM
# --------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# --------------------------------
# GIVE TOOLS TO THE AGENT
# --------------------------------
tools = [
    get_weather,
    calculate,
    get_time
]


# --------------------------------
# CREATE AGENT
# --------------------------------
agent = create_agent(
    model=llm,
    tools=tools
)


# --------------------------------
# USER INPUT
# --------------------------------
question = input("You: ")


# --------------------------------
# RUN AGENT
# --------------------------------


result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    }
)


# --------------------------------
# PRINT FINAL ANSWER
# --------------------------------

print("\nAI:", result["messages"][-1].content)

# Agent is a system where the LLM can can decide what action/tool to take, execuit the tool,
#  look at the result, and continue deciding until it can produce the final answer.


# Agent = system that connects the LLM + tools + execution loop
