# ChatOpenAI documentation - https://python.langchain.com/docs/integrations/chat/openai/
# Parameters - https://api.python.langchain.com/en/latest/chat_models/langchain_community.chat_models.openai.ChatOpenAI.html

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# llm1 = ChatOpenAI(model="gpt-4o-mini")
# result1 = llm1.invoke("What is the capital of USA?")
# print(result1)

# returns a more descriptive answer, and actual answer is in "content" key.

# llm2 = ChatOpenAI(model="gpt-4o-mini")
# result2 = llm2.invoke("What is the capital of USA?")
# print(result2.content)

# llm3 = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
# result3 = llm3.invoke("What is bigger? 9.9 or 9.11")
# print(result3.content)

# returns 9.9, good!

# llm4 = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
# result4 = llm4.invoke("How many r's in the word 'strawberry'?")
# print(result4.content)

# returns 2 again, huh? 4o-mini is a newer model and supposed to handle all this.

# let's try resampling 2 more times (as it's all a game of probability) with temp=0.1

# llm5 = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
# result5 = llm5.invoke("How many r's in the word 'strawberry'?")
# print(result5.content)

# llm6 = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
# result6 = llm6.invoke("How many r's in the word 'strawberry'?")
# print(result6.content)

# llm5 returns 2, and llm6 returns 3. Well, resampling works, and from here you go on to reinformcement learning!

# llm7 = ChatOpenAI(model="gpt-4o-mini", temperature=1.8, max_tokens=100)
# result7 = llm7.invoke("Write a short story about a cat")
# print(result7.content)

# returns a bit gibberish and has a Hindi alphabet, 4o-mini shouldnt be doing this but maybe temperature is too high?

# llm8 = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_completion_tokens=100)
# result8 = llm8.invoke("Write a short story about a cat")
# print(result8.content)

# returns a perfect answer now, its just cut off in between.

llm9 = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=100) # max_tokens is deprecated (still works but just points to latter), use max_completion_tokens instead
result9 = llm9.invoke("Write a short story about a cat")
print(result9.content)

# same thing. 













