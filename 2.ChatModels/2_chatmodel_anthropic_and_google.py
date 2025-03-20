# I don't have the API key for these as I already bought openai one and have already experimented with that,
#  so I can't run this code but anyway good to know the syntax.

from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-3-5-sonnet-20241022')
result = model.invoke('What is the capital of India')
print(result.content)

# ----------------------------------------

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
result = model.invoke('What is the capital of India')
print(result.content)

# ----------------------------------------





