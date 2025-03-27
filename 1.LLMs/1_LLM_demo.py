from langchain_openai import OpenAI # integration package between OpenAI and LangChain
from dotenv import load_dotenv # to load environment variables from .env file

load_dotenv() # load environment variables from .env file

# ref - https://openai.com/api/pricing/

llm = OpenAI(model="gpt-3.5-turbo-instruct") # initialize the OpenAI model 
# 3.5-turbo-instruct is a text-completion model only, not chat model, https://python.langchain.com/docs/integrations/llms/openai/

result = llm.invoke("What is the capital of India?") # invoke the model

print(result)

# These LLMs are not much used these days though, instead, ChatModels are used.

# more ref to documentation

# LangChain Documentation for LLMs:
# Main LangChain LLMs documentation: https://python.langchain.com/docs/introduction/, https://python.langchain.com/docs/concepts/#llms
# LangChain OpenAI integration: https://python.langchain.com/docs/integrations/llms/openai

# OpenAI API reference: https://platform.openai.com/docs/api-reference
# GPT-3.5-turbo-instruct model specifics: https://platform.openai.com/docs/models/gpt-3-5-turbo-instruct

# Transitioning to Chat Models: https://python.langchain.com/docs/modules/model_io/chat/



# Trying other parameters (https://api.python.langchain.com/en/latest/llms/langchain_openai.llms.base.OpenAI.html)

llm1 = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9, max_tokens=100)  # temperature = 1 is default
result1 = llm1.invoke("Write a haiku about cats") 
print(result1)

# returns Soft fur and sharp claws, Graceful paws and curious eyes, Cats purr, nap, and play.

llm2 = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.1, max_tokens=100) 
result2 = llm2.invoke("Write a haiku about cats") 
print(result2)

# returns Furry feline friend, Purring softly in my lap, Contentment in fur



# Streaming results

llm4 = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.3, max_tokens=100) 
result4 = llm4.stream("What is bigger? 0.9 or 0.11") 
for chunk in result4:
    print(chunk, end="--", flush=True)

# returns --0--.--11-- is-- bigger--.----
# we see that it runs into the classic LLM problem of hallucinations, probably because 3.5 is an older model.

llm5 = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.3, max_tokens=100) 
result5 = llm5.invoke("How many r's in the word 'strawberry'?") 
print(result5)

# returns 2, problem again.
