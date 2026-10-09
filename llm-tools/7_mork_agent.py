from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# 1. Setup in-memory history storage
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 2. Create a Mock Agent that mimics your agent's behavior
# It takes the input messages and appends its own tool and final responses
def mock_agent(input_dict):
    input_msgs = input_dict.get("messages", [])
    
    # Simulate the agent returning the full conversation state
    return {
        "messages": input_msgs + [
            AIMessage(content="", additional_kwargs={"tool_calls": [{"id": "call_1", "function": {"name": "get_weather", "arguments": '{"city": "Hyderabad"}'}}]}),
            ToolMessage(content="30°C and sunny", tool_call_id="call_1"),
            AIMessage(content="Hyderabad is 30°C and sunny.")
        ]
    }

agent = RunnableLambda(mock_agent)

# 3. Wrap with history (reproducing your exact configuration)
agent_with_history = RunnableWithMessageHistory(
    agent,
    get_session_history,
    input_messages_key="messages",
    output_messages_key="messages"
)

# 4. First Invocation
print("--- RUNNING FIRST INVOCATION ---")
agent_with_history.invoke(
    {"messages": [HumanMessage(content="What is the weather in Hyderabad?")]},
    config={"configurable": {"session_id": "user_1"}}
)

# 5. Print the stored history
history = get_session_history("user_1")
print("\n--- ACTUAL STORED HISTORY AFTER INVOCATION 1 ---")
for i, msg in enumerate(history.messages):
    print(f"{i+1}. {msg.__class__.__name__}: {msg.content or '[Tool Call]'}")