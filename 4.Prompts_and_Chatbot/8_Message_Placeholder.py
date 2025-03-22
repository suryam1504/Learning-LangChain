# To retrieve older conversation from a database
# Ref - https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# 1.

# In addition to Human/AI/Tool/Function messages,
# you can initialize the template with a MessagesPlaceholder
# either using the class directly or with the shorthand tuple syntax:

template = ChatPromptTemplate([
    ("system", "You are a helpful AI bot."),
    ("placeholder", "{conversation}") # Means the template will receive an optional list of messages under the "conversation" key
    # Equivalently:
    # MessagesPlaceholder(variable_name="conversation", optional=True)
])

prompt_value = template.invoke(
    {
        "conversation": [
            ("human", "Hi!"),
            ("ai", "How can I assist you today?"),
            ("human", "Can you make me an ice cream sundae?"),
            ("ai", "No.")
        ]
    }
)

print(prompt_value)


# 2.


# chat template
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = []

# load chat history
with open('4.Prompts_and_Chatbot/8.1_chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

print("---------------------------")

# create prompt
prompt = chat_template.invoke({'chat_history':chat_history, 'query':'Where is my refund'})

print(prompt)