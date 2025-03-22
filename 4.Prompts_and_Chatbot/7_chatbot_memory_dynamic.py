# Dynamic prompts for a list of messages using ChatPromptTemplate
# Ref - https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain':'cricket','topic':'Wide ball'})

print(prompt)

# Do NOT put SystemMessage, HumanMessage, AIMessage in ChatPromptTemplate to define system and human messages, that's just wrong acc to LangChain syntax and the code runs but the placeholders are not actually filled with values, pass tuples like above instead 

# ChatPromptTemplate.from_message([]) also does the same thing