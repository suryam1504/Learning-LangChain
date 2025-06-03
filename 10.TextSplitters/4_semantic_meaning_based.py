# see text example below to understand the type of texts where this is helpful

# how this works: emb of every sentence is calculated (say s1 s2 s3.............), then cosine similarity calculated between s1 and s2, if score is high they are merged, then between s2 ans s3, if high then merged (i guess s1 is also medged with all this), b/w s3 and s4, if score low then this is decided as the breaking point

# note: langchain documentation (https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb) says this tho - At a high level, this splits into sentences, then groups into groups of 3 sentences, and then merges one that are similar in the embedding space.

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation", # sd is calculated of cosine similarity scores between all adjacent/consecutive sentences. (there are other methods too, like percentiles or interquartile ranges, etc.)
    breakpoint_threshold_amount=2 # if any of the similarity score is more or less than 2 std dev, that would be the breaking point
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.


Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)

# >>>

# 2
# [Document(metadata={}, page_content='\nFarmers were working hard in the fields, preparing the soil and planting seeds for the next season.'), Document(metadata={}, page_content='The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams. Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety. ')]

# we would have ideally liked to see 3 docs, one of farmers and sun was bright, then of IPL, then of terrorism. So these results are not perfect, mainly as this concept is also new and all this is still in experimental state (hence the from langchain_experimental.text_splitter import SemanticChunker!), hence this is not used a lot as of now, but potential!




# see this detailed thing i found from langchain documentation - https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb