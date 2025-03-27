# Getting a summary and questions sequentially from a topic 

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Give me a summary on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate 3 questions from the summary:\n {summary}",
    input_variables=["summary"]
)

chain = prompt1 | llm | parser | prompt2 | llm | parser

result = chain.invoke({'topic':"History of Ambulance"})

print(result)

# 1. What were some of the methods used to transport the sick or injured in ancient civilizations, and how did these methods change during the Middle Ages?

# 2. How did the introduction of horse-drawn ambulances in the 19th century impact the efficiency of transporting patients to hospitals, and what was notable about the first city ambulance service?

# 3. In what ways did the advent of motorized ambulances and the experiences of World War I and II contribute to advancements in ambulance technology and emergency medical care?  

chain.get_graph().print_ascii()

#      +-------------+       
#      | PromptInput |
#      +-------------+
#             *
#             *
#             *
#     +----------------+
#     | PromptTemplate |
#     +----------------+
#             *
#             *
#             *
#       +------------+
#       | ChatOpenAI |
#       +------------+
#             *
#             *
#             *
#    +-----------------+
#    | StrOutputParser |
#    +-----------------+
#             *
#             *
#             *
# +-----------------------+
# | StrOutputParserOutput |
# +-----------------------+
#             *
#             *
#             *
#     +----------------+
#     | PromptTemplate |
#     +----------------+
#             *
#             *
#             *
#       +------------+
#       | ChatOpenAI |
#       +------------+
#             *
#             *
#             *
#    +-----------------+
#    | StrOutputParser |
#    +-----------------+
#             *
#             *
#             *
# +-----------------------+
# | StrOutputParserOutput |
# +-----------------------+