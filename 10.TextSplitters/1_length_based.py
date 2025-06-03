# user decides the length/size of chunks that will be created, either on character or word/token based
# https://chunkviz.up.railway.app/

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

text = """
Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalise to unseen data, and thus perform tasks without explicit instructions.[1] Within a subdiscipline in machine learning, advances in the field of deep learning have allowed neural networks, a class of statistical algorithms, to surpass many previous machine learning approaches in performance.
"""

splitter = CharacterTextSplitter(
    chunk_size=200, # break after every 200 character
    chunk_overlap=0, # no of characters to be shared between 2 chunks, helps mitigate the abrupt cutoff, for RAG based applications, it is said that 10 to 20% of the chunk size is a good estimate for chunk overlap
    separator=''
)

result1 = splitter.split_text(text)
print(result1)

# ['Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalise to unseen data, and thu', 's perform tasks without explicit instructions.[1] Within a subdiscipline in machine learning, advances in the field of deep learning have allowed neural networks, a class of statistical algorithms, to', 'surpass many previous machine learning approaches in performance.']

# its a list, but we see how words are cut off in between bcoz we split using characters, this is not good as contextual meaning will be lost in the embeddings


# now, connecting docloader and textsplitter workflow

loader = PyPDFLoader('10.TextSplitters\dl-curriculum.pdf')

docs = loader.load()

result = splitter.split_documents(docs)

print(result)
print(result[0]) # 1st chunk, has metadata and page conent
print(result[1].page_content)


