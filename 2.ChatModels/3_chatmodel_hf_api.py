# Langchain HF Doc - https://python.langchain.com/v0.1/docs/integrations/chat/huggingface/
# https://api.python.langchain.com/en/latest/community/llms/langchain_community.llms.huggingface_hub.HuggingFaceHub.html

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv()

print(os.getenv("HUGGINGFACEHUB_API_TOKEN")) # it should print the actual token if everything is fine

# to make LLM which we want to use
llm = HuggingFaceEndpoint(
    repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation")

model = ChatHuggingFace(llm = llm)

result = model.invoke("What is the capital of India")
query = model.invoke("What is 10 times 3?")

print(result.content)
print(query.content)
# returns wrong answer and a lot of unrelated information, but only 1.1B model so expected.


llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    task = "text-generation",
    max_new_tokens=100,
    temperature=0.2)
model = ChatHuggingFace(llm = llm)
result = model.invoke("What is the capital of India")
print(result.content)


llm1 = HuggingFaceEndpoint(repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", task = "text-generation",)
model1 = ChatHuggingFace(llm = llm1, temperature = 1.3, max_completion_tokens = 100)
result1 = model1.invoke("Give me names of 3 colors and what will I get if I mix them")
print(result1.content)











# try more models from HF (llama3, mistral, deepseek, etc) and with different parameters and prompts
# check this for assitant user thing - https://python.langchain.com/v0.1/docs/integrations/chat/huggingface/
