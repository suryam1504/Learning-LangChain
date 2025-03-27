# Simple Chain

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser # string output parser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables = "topic" # or input_variables = ["topic"]
)

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({'topic':"eggs"})

print(result)



# Now,to visualize the chain

chain.get_graph().print_ascii()  # needs pip install grandalf

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