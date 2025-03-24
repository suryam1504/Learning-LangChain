# Certain LLMs (like closed source ones like OpenAI models) have the ability to give structured outputs (let's call them "can llms") which we can call with with_structured_output, but other LLMs might not have this ability ("can't llms"). So to deal with "can't llms", we use output parsers. This means output parsers can anyway work with "can llms" models too. We will look into 4 output parsers out of many which exist: string output parser, json output parser, structured output parser, and pydantic output parser.

# 1. String Output Parser - returns LLM output as a string, i.e. usually there are a lot of parameters in the output and usually we just get result.content, but with this we wouldnt have to do that, and this can be effective when we need to pass the output to another LLM and create chains.

# let's say we want to talk to LLM 2 times, first to generate a detailed report on a topic, and then to generate a 5-line summary of the report. Now comparing result.content usage vs string output parser usage:

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

# 1st prompt - detailed report

template1 = PromptTemplate(
    template="""Generate a detailed report on the topic: {topic}""",
    input_variables=["topic"]
)

# 2nd prompt - summary

template2 = PromptTemplate(
    template="""Generate a 5-line summary of the report: {report}""",
    input_variables=["report"]
)

# using result.content method

prompt1 = template1.invoke({"topic": "Black Hole"})

result1 = model.invoke(prompt1)

print(result1.content)
print("--------------------------------")

prompt2 = template2.invoke({"report": result1.content})

result2 = model.invoke(prompt2)

print(result2.content)

     

# using string output parser method

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

# this chain creates a pipeline, what it does is:
# i. template1: takes in a topic and generates template1
# ii. model: takes in the template1 and generates a report (which has a lot of parameters)
# iii. parser: takes in this output and parses it into a string (report) (essentially extracting result.content)
# iv. template2: takes in the string (report) and generates template2
# v. model: takes in the template2 and generates a string (summary) (which again has a lot of parameters)
# vi. parser: takes in this output and parses it into a string (summary) (essentially extracting result.content)

result = chain.invoke({"topic": "Black Hole"})

print(result)




