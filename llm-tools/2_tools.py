from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage


# =====================================================
# 1. Create a tool
# =====================================================

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a city.
    """

    # Dummy weather data for learning
    return f"The weather in {city} is 30°C and sunny."


# =====================================================
# 2. Create LLM
# =====================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# =====================================================
# 3. Bind the tool to the LLM
# =====================================================

llm_with_tools = llm.bind_tools([
    get_weather
])


# =====================================================
# 4. Get user question
# =====================================================

question = input("You: ")


# =====================================================
# 5. Send question to LLM
# =====================================================

response = llm_with_tools.invoke(question)


print("\n--- LLM Response ---")
print(response)


# =====================================================
# 6. Check whether LLM requested a tool
# =====================================================

if response.tool_calls:

    tool_call = response.tool_calls[0]

    print("\n--- Tool Call ---")
    print("Tool:", tool_call["name"])
    print("Arguments:", tool_call["args"])


    # =================================================
    # 7. Execute the requested tool
    # =================================================

    if tool_call["name"] == "get_weather":

        tool_result = get_weather.invoke(
            tool_call["args"]
        )

        print("\n--- Tool Result ---")
        print(tool_result)


        # =============================================
        # 8. Send tool result back to LLM
        # =============================================

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


        # =============================================
        # 9. Final answer
        # =============================================

        print("\n--- Final Answer ---")
        print(final_response.content)


else:

    # LLM answered without using a tool
    print("\n--- Final Answer ---")
    print(response.content)




# ------------------------------------------------------------------------------------------

# [
#     {
#         "name": "get_weather",
#         "args": {
#             "city": "Hyderabad"
#         }
#     }
# ]