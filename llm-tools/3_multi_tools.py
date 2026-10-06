from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage


# =====================================================
# TOOL 1
# =====================================================

@tool
def get_weather(city: str) -> str:
    """
    Get the weather for a city.
    """

    # Dummy data for learning
    weather_data = {
        "hyderabad": "30°C and sunny",
        "delhi": "27°C and cloudy",
        "mumbai": "29°C and humid"
    }

    return weather_data.get(
        city.lower(),
        f"Weather information for {city} is not available."
    )


# =====================================================
# TOOL 2
# =====================================================

@tool
def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression.
    """

    try:
        result = eval(expression)
        return str(result)

    except Exception:
        return "Could not calculate the expression."


# =====================================================
# TOOL 3
# =====================================================

@tool
def get_current_time(city: str) -> str:
    """
    Get the current time for a city.
    """

    # Dummy value for learning
    return f"The current time in {city} is 10:30 AM."


# =====================================================
# CREATE LLM
# =====================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# =====================================================
# GIVE MULTIPLE TOOLS TO LLM
# =====================================================

tools = [
    get_weather,
    calculate,
    get_current_time
]

llm_with_tools = llm.bind_tools(tools)


# =====================================================
# USER QUESTION
# =====================================================

question = input("You: ")


# =====================================================
# FIRST LLM CALL
# =====================================================

response = llm_with_tools.invoke(question)

print("\n--- LLM Response ---")
print(response)


# =====================================================
# CHECK TOOL CALL
# =====================================================

if response.tool_calls:

    tool_call = response.tool_calls[0]

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    print("\n--- Selected Tool ---")
    print(tool_name)

    print("\n--- Arguments ---")
    print(tool_args)


    # =================================================
    # FIND THE CORRECT TOOL
    # =================================================

    tool_map = {
        "get_weather": get_weather,
        "calculate": calculate,
        "get_current_time": get_current_time
    }

    selected_tool = tool_map[tool_name]


    # =================================================
    # EXECUTE TOOL
    # =================================================

    tool_result = selected_tool.invoke(tool_args)

    print("\n--- Tool Result ---")
    print(tool_result)


    # =================================================
    # SEND RESULT BACK TO LLM
    # =================================================

    messages = [
        {
            "role": "user",
            "content": question
        },

        response,

        ToolMessage(
            content=tool_result,
            tool_call_id=tool_call["id"]
        )
    ]


    final_response = llm_with_tools.invoke(messages)


    # =================================================
    # FINAL ANSWER
    # =================================================

    print("\n--- Final Answer ---")
    print(final_response.content)


else:

    print("\n--- Final Answer ---")
    print(response.content)