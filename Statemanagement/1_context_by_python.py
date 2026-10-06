
# A simple solution is: keep recent messages + summarize older messages.



from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

chat_history = []

MAX_MESSAGES = 6


while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    # Add user message
    chat_history.append(
        HumanMessage(content=question)
    )

    # If history becomes too large
    if len(chat_history) > MAX_MESSAGES:

        # Take old messages
        old_messages = chat_history[:-4]

        # Create text from old messages
        old_text = "\n".join(
            f"{type(m).__name__}: {m.content}"
            for m in old_messages
        )

        # Ask LLM to summarize old conversation
        summary_response = llm.invoke([
            HumanMessage(
                content=f"""
Summarize this conversation briefly.
Keep important facts and decisions.

Conversation:
{old_text}
"""
            )
        )

        summary = summary_response.content

        # Keep only recent messages
        chat_history = [
            HumanMessage(
                content=f"Previous conversation summary: {summary}"
            )
        ] + chat_history[-4:]

    # Send context to LLM
    response = llm.invoke(chat_history)

    print("AI:", response.content)

    # Add AI response
    chat_history.append(
        AIMessage(content=response.content)
    )