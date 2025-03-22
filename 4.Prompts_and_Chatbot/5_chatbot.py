# Simple chatbot which has no memory and no context

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("AI: Goodbye!")
        break
    result = model.invoke(user_input)
    print("AI: ", result.content)


# Simple chatbot with memory and context, and explicit mention of the User and AI roles in chat history for the LLM to follow (and not get confused as conversation gets longer)

chat_history = []

while True:
    user_input = input("You: ")
    chat_history.append(f"You: {user_input}")
    if user_input.lower() == "exit":
        print("AI: Goodbye!")
        break
    result = model.invoke(chat_history) # invoke function here is flexible enough to take in either a single query or list of messages
    chat_history.append(f"AI: {result.content}")
    print("AI: ", result.content)

print(chat_history)

# The way we defined dictionary of messages manually is not the best way to do it, Langchain already identified this problem and made an efficient syntax, see 6_chatbot_memory.py for a better way
