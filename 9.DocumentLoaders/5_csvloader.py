# every row becomes a document object

from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='9.DocumentLoaders/Social_Network_Ads.csv')

docs = loader.load()

print(len(docs)) # 400
print(docs[1]) # 2nd row