# Managing converstion state with ConversationBufferMemory 

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# 1. Create memory
memory = ConversationBufferMemory(
    return_messages=True
)


# 2. Create ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    
    # System message
    (
        "system",
        "You are a helpful AI assistant. "
        "Answer the user clearly and simply."
    ),

    # Previous conversation will be inserted here
    MessagesPlaceholder(
        variable_name="history"
    ),

    # Current user question
    (
        "human",
        "{input}"
    )
])


# 3. Create ConversationChain
chat = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt
)


# 4. Chat loop
while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    response = chat.invoke({
        "input": question
    })

    print("AI:", response["response"])