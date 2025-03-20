from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# put which emb model to use (https://platform.openai.com/docs/guides/embeddings) and how many dimensions to use (1536 for small, 3072 for large)
embedding = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=32)

result = embedding.embed_query("Delhi is the capital of India")

print(str(result))

# [0.16083911061286926, 0.09522069990634918, 0.1782715618610382, 0.08200930804014206, -0.2175220102071762, 0.20425580441951752, -0.09483696520328522, 
# 0.3076445460319519, -0.10053814947605133, 0.003926414996385574, 0.15437045693397522, 0.2672977149486542, -0.24690502882003784, 0.019049622118473053, 
# 0.05583320930600166, -0.26422783732414246, -0.2541411519050598, 0.09171228110790253, 0.13978859782218933, 0.19800642132759094, 0.36991897225379944, 
# 0.08743639290332794, 0.19296307861804962, 0.12498744577169418, 0.3153192102909088, 0.14581868052482605, -0.061342522501945496, -0.00967556331306696, 
# 0.1622644066810608, 0.08332496136426926, -0.07724004983901978, -0.036728765815496445]



# how to convert multiple queries into embeddings

documents = [
    "Delhi is the capital of India",
    "Is Apple a fruit or a company?",
    "Lugano is a city in Switzerland",
]

result1 = embedding.embed_documents(documents)

print(str(result1))


