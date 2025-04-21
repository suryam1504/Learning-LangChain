# to load pdf files - but very simple textual content ones, not complex (like a pdf with scanned images, etc.), as it use pypdf library under the hood which is not the best thing out there

from langchain_community.document_loaders import PyPDFLoader # works on a page by page basis

loader = PyPDFLoader('9.DocumentLoaders\dl-curriculum.pdf') # has 23 pages

docs = loader.load()

print(len(docs)) # 23

print(docs[0].page_content) # actul textual content of 1st page in pdf
print(docs[0].metadata)
print(docs[1].page_content)

# more pdf loaders available in langchain (https://python.langchain.com/docs/integrations/document_loaders/), like
# 1. PDFPLumberLoader - pdfs with tables/columns
# 2. UnstructuredPDFLoader or AmazonTextractorPDFLoader - pdfs with scanned images
# 3.  PyMuPDFLoader - need layout and image 
 