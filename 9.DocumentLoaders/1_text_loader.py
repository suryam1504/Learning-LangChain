# reads plain text files (.txt) and converts them to LangChain Document objects.

# diff types of doc loaders - https://python.langchain.com/docs/integrations/document_loaders/

from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('9.DocumentLoaders\cricket.txt', encoding='utf-8')

docs = loader.load()

print(docs) # has page_content and metadata, which can be extracted seprately
print(type(docs))  # <class 'list'>, list of documents
print(len(docs))
print(docs[0])
print(type(docs[0])) # <class 'langchain_core.documents.base.Document'>
print(docs[0].page_content)
print(docs[0].metadata)


# now usual LLM stuff

parser = StrOutputParser()

model = ChatOpenAI(model="gpt-4o-mini")

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

chain = prompt | model | parser

print(chain.invoke({'poem':docs[0].page_content}))

# >>> The poem celebrates the spirit, thrill, and cultural significance of cricket, portraying it as more than just a sport—it's a communal experience that spans generations and transcends borders. It begins by setting the scene of the game, depicting the vibrant atmosphere of both local fields and grand stadiums. The excitement builds with the toss of the coin, as players prepare for the contest, facing the contrasting roles of batting and bowling.

# As the game unfolds, the poem vividly describes the tension and joy of scoring runs, the skill involved in bowling and batting, and the deep emotional connection fans feel toward the sport. It highlights memorable moments, from glorious victories to heartbreaking losses, and the pivotal interactions between players, fans, and technology like the DRS system.

# The narrative weaves through the joy of playing cricket, the camaraderie it fosters, and the history it carries, mentioning legendary players and their contributions to the game. It also emphasizes cricket's ability to unite people across different cultures and backgrounds, becoming a shared passion that ignites hope and pride.

# Ultimately, the poem encapsulates cricket's timelessness and its capacity to evoke cherished memories, encouraging future generations to cherish and participate in this beloved game. It concludes with a heartfelt call to continue the tradition of cricketing, celebrating its enduring magic.