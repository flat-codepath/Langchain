from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage


# -------------------------
# TOOL 1
# -------------------------
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    
    # Fake weather data for learning
    weather_data = {
        "hyderabad": "30°C and sunny",
        "delhi": "32°C and clear",
        "mumbai": "29°C and cloudy",
    }

    return weather_data.get(
        city.lower(),
        f"Weather information is not available for {city}"
    )


# -------------------------
# TOOL 2
# -------------------------
@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""

    try:
        result = eval(expression)
        return str(result)

    except Exception:
        return "Invalid mathematical expression"


# -------------------------
# TOOL 3
# -------------------------
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


# -------------------------
# CREATE LLM
# -------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# Give ALL tools to the LLM
tools = [
    get_weather,
    calculate,
    get_time
]

llm_with_tools = llm.bind_tools(tools)


# -------------------------
# USER QUESTION
# -------------------------
question = input("You: ")

messages = [
    {
        "role": "user",
        "content": question
    }
]


# -------------------------
# TOOL LOOP
# -------------------------
while True:

    response = llm_with_tools.invoke(messages)

    # Add LLM response to conversation
    messages.append(response)

    # Did the LLM request any tools?
    if not response.tool_calls:

        print("\nAI:", response.content)
        break


    # LLM requested one or more tools
    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print("\nLLM decided to call:", tool_name)
        print("Arguments:", tool_args)


        # Find the requested tool
        if tool_name == "get_weather":

            result = get_weather.invoke(tool_args)

        elif tool_name == "calculate":

            result = calculate.invoke(tool_args)

        elif tool_name == "get_time":

            result = get_time.invoke(tool_args)

        else:

            result = "Unknown tool"


        print("Tool result:", result)


        # Send tool result back to LLM
        messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            )
        )




# Try this question
# You: What is the weather in Hyderabad and calculate 25 * 4

# You should see something similar to:

# LLM decided to call: get_weather
# Arguments: {'city': 'Hyderabad'}

# Tool result: 30°C and sunny

# LLM decided to call: calculate
# Arguments: {'expression': '25 * 4'}

# Tool result: 100

# AI: The weather in Hyderabad is 30°C and sunny, and 25 * 4 is 100.


# Tool calling + loop = the LLM can perform multiple actions before producing the final answer.


# Agent = system that connects the LLM + tools + execution loop

# Agent is a system where the LLM can can decide what action/tool to take, execuit the tool,
#  look at the result, and continue deciding until it can produce the final answer.

# Agent is a system where the LLM can decide what action/tools to take, and execute the tools and look at the results and continue looping until it can produce the final answer.