# uses concept of Recursive Character Text Splitter, which presumes a structure in a text that there's a paragraph first, then a line, then words, and then nothing, and accordingly tries to split a text by, say, "\n\n" character first (para), then "\n" (line), then " " (a soace, for words), and "" (nothing). Essentialy it tries as much as possible to avoid abrupt cut-offs.

from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalise to unseen data, and thus perform tasks without explicit instructions.

Within a subdiscipline in machine learning, advances in the field of deep learning have allowed neural networks, a class of statistical algorithms, to surpass many previous machine learning approaches in performance.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100, # this is still character based
    chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)

# 6
# ['Machine learning (ML) is a field of study in artificial intelligence concerned with the development', 'and study of statistical algorithms that can learn from data and generalise to unseen data, and', 'thus perform tasks without explicit instructions.', 'Within a subdiscipline in machine learning, advances in the field of deep learning have allowed', 'neural networks, a class of statistical algorithms, to surpass many previous machine learning', 'approaches in performance.']

# mostly this is used, ass the way of these chunk breaks is also how usually context is broken in real life writings