
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory


# --------------------------------
# TOOL 1: WEATHER
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
# TOOL 2: CALCULATOR
# --------------------------------
@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""

    try:
        # Demo only: eval can execute arbitrary Python code.
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)

    except Exception:
        return "Invalid mathematical expression"


# --------------------------------
# TOOL 3: TIME
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
# CREATE AGENT
# --------------------------------
tools = [get_weather, calculate, get_time]

agent = create_agent(
    model=llm,
    tools=tools
)


# --------------------------------
# CHAT HISTORY STORAGE
# --------------------------------
store = {}


def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


# --------------------------------
# CHAT PROMPT TEMPLATE
# --------------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. "
        "Use the available tools when needed. "
        "Answer clearly and simply."
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])


# --------------------------------
# ADAPT PROMPT OUTPUT FOR THE AGENT
# --------------------------------
def prepare_agent_input(prompt_value):
    return {"messages": prompt_value.messages}


# --------------------------------
# EXTRACT THE FINAL AGENT RESPONSE
# --------------------------------
def extract_final_answer(agent_result):
    return {"answer": agent_result["messages"][-1]}


# --------------------------------
# PIPE OPERATOR
# --------------------------------
chain = (
    prompt
    | RunnableLambda(prepare_agent_input)
    | agent
    | RunnableLambda(extract_final_answer)
)


# --------------------------------
# WRAP WITH MESSAGE HISTORY
# --------------------------------
chat = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
    output_messages_key="answer"
)


# --------------------------------
# CHAT LOOP
# --------------------------------
while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    result = chat.invoke(
        {"question": question},
        config={
            "configurable": {
                "session_id": "user_1"
            }
        }
    )

    print("\nAI:", result["answer"].content)






# The One Trade-off to Keep in Mind
# By slicing [-1], your InMemoryChatMessageHistory will now only store the Human inputs and the final AI text responses. It will discard the tool calls and tool results from the memory.

# The Good: Your chat history stays incredibly clean (Human -> AI -> Human -> AI). You don't waste tokens feeding raw tool JSON back into the LLM on subsequent turns.

# The Bad: On the next turn, the LLM will remember what it told the user (e.g., "Hyderabad is 30°C"), but it will not remember how it found out (it won't see the tool execution in its history).

# For 95% of standard chatbots, discarding the tool messages from the long-term memory is exactly what you want to do. Your pipeline is a standard and highly effective LangChain Expression Language (LCEL) pattern.