from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Magnus Carlsen is a Norwegian chess grandmaster who was the World Chess Champion from 2013 to 2023.",
    "Hikaru Nakamura is an American chess grandmaster and 5 times US Chess Champion.",
    "Fabiano Caruana is an Italian chess grandmaster who came close to winning the World Chess Championship in 2018.",
    "Ding Liren is a Chinese chess grandmaster who became the World Chess Champion in 2021 after defeating Nepo.",
    "Gukesh Dommaraju is an Indian chess grandmaster who is currently the World Chess Champion."]

query = 'tell me about ding'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

print(cosine_similarity([query_embedding], doc_embeddings))
scores = cosine_similarity([query_embedding], doc_embeddings)[0] # put query_embedding in a 2d array

print("-----------------------")

print(list(enumerate(scores)))
print(sorted(list(enumerate(scores)),key=lambda x:x[1])[-1])
index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1] # sorted(...,key=lambda x:x[1]) means sort the list by the second element of the tuple, which is similarity scores

print("-----------------------")

print(query)
print(documents[index])
print("similarity score is:", score)

# tell me about ding
# Ding Liren is a Chinese chess grandmaster who became the World Chess Champion in 2021 after defeating Nepo.
# similarity score is: 0.4148600493276643