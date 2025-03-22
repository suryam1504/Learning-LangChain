# psuedo code of how different types of messages in LangChain are defined

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage # these are also static prompts as of now, see 7_chatbot_memory_dynamic.py for dynamic prompts for a list of messages using ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about Zebra')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)



# we now integrate this into our original chatbot from 5_chatbot.py

chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print(chat_history)

# run this and see how it stores all the information form system, user, and AI in a much more systematic manner