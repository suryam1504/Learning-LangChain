# to load multiple documents from a directory/folder

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='D:/LLMs/Learning-LangChain/9.DocumentLoaders/Books',
    glob='*.pdf', # load all pdf files inside Book folder
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(len(docs)) # 435, addition of all pages in the 3 pdfs

print(docs[0].page_content) # content of 1st page of 1st pdf
print(docs[0].metadata)

for document in docs:
    print(document.metadata)

# this takes time to run, as we are loading all the 3 pdfs together. Hence a folder with 100 pdfs with a lot of pages will take a lot of time.
# Load() uses Eager Loading, meaning loading everything at once and loads all documents immediately into memory. Returns a list of document objects.


# Hence, Lazy Load - loads on demand, docs are not all loaded at once, they are fetched one at a time as needed, returns a generator of document objects.

# run above and this and see differnce, this keeps printing metadata and then removes it from memory and mvoes onto next one
docs1 = loader.lazy_load()

for document in docs1:
    print(document.metadata)



