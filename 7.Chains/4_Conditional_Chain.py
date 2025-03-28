# Based on pos or neg customer feedback, we will return a response

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini") # we can use different models for different parts of the chain
parser = StrOutputParser() # this should not be used as it might return "this feedback is negative" instead of strictly pos or neg

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Give the sentiment of the customer feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the customer feedback as positive or negative: \n {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | llm | parser2

result = classifier_chain.invoke({"feedback":"The battery life is too short!"}).sentiment

#print(result)

# "negative"

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

# this is like if else statement
branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | llm | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | llm | parser),
    RunnableLambda(lambda x: "could not find sentiment") # default chain
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a beautiful phone'}))

chain.get_graph().print_ascii()


# Thank you so much for your positive feedback! We're thrilled to hear that you had a great experience. Your support means a lot to us, and we’re committed to continuing to provide excellent service. If you have any suggestions or further thoughts, we’d love to hear them!
#     +-------------+      
#     | PromptInput |      
#     +-------------+      
#             *
#             *
#             *
#    +----------------+    
#    | PromptTemplate |
#    +----------------+
#             *
#             *
#             *
#      +------------+
#      | ChatOpenAI |
#      +------------+
#             *
#             *
#             *
# +----------------------+
# | PydanticOutputParser |
# +----------------------+
#             *
#             *
#             *
#        +--------+
#        | Branch |
#        +--------+
#             *
#             *
#             *
#     +--------------+
#     | BranchOutput |
#     +--------------+
