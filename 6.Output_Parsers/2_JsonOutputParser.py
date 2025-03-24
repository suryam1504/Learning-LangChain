from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

parser = JsonOutputParser()


# how this works

template1 = PromptTemplate(
    template="Give me name, age, and gender of any celebrity. \n {format_instruction}",
    input_variables=[],  # nothing here as user doesn't enter any input in the prompt in this case
    partial_variables={'format_instruction': parser.get_format_instructions()} # we use the parser's, which is a json output parser, .get_format_instructions() method to get the format instructions, which puts "Return a JSON object." in the prompt automatically.
)

prompt1 = template1.format() # can also use template1.invoke({}) method with empty dictionary as no input variables are present in the prompt

print(prompt1)
print("--------------------------------")

# output
# Give me name, age, and gender of any celebrity. 
#  Return a JSON object.

template2 = PromptTemplate(
    template="Give me name, age, and gender of {celebrity}. \n {format_instruction}",
    input_variables=['celebrity'],  
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt2 = template2.format(celebrity="Natalie Portman") 

print(prompt2)
print("--------------------------------")

# Give me name, age, and gender of Natalie Portman.
#  Return a JSON object.

# result1 = model.invoke(prompt1)
# final_result1 = parser.parse(result1.content)
# type_of_result1 = type(final_result1)
# print(result1)  # this is the raw output from the model about Taylor Swift (another sampling gave TS again with age 34 tho, and another gave Emma Watson)
# print(final_result1) # {'name': 'Taylor Swift', 'age': 33, 'gender': 'Female'}
# print(type_of_result1) # <class 'dict'>

# print("--------------------------------")

# result2 = model.invoke(prompt2)
# final_result2 = parser.parse(result2.content)
# type_of_result2 = type(final_result2)
# print(result2)
# print(final_result2) # {'name': 'Natalie Portman', 'age': 42, 'gender': 'Female'}
# print(type_of_result2) # <class 'dict'>




# in use, and effectively using chains again

template = PromptTemplate(
    template='Give me 5 facts about {topic}. \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'Walmart'})

print(result)

# {'WalmartFacts': [{'fact': 'Walmart was founded in 1962 by Sam Walton in Rogers, Arkansas.'}, {'fact': "As of 2023, Walmart is the world's largest retailer, operating over 10,500 stores in more than 20 countries."}, {'fact': 'Walmart employs over 2.3 million people globally, making it one of the largest employers in the world.'}, {'fact': "Walmart's mission statement is 'We save people money so they can live better.'"}, {'fact': 'Walmart has committed to achieving 100% renewable energy in its global operations by 2035.'}]}

# here we notice that the output is a dictionary with a key 'WalmartFacts' and a list of dictionaries as its value. Each dictionary in the list has a key 'fact' and a value which is a string. This is what the LLM decided on its own.

# But what if we wanted {fact1: ..., fact2: ..., fact3: ..., fact4: ..., fact5: ...} format? Unfornutaly, the JsonOutputParser does not support this and we cannot enforce our own schema. We can try being more specific in the template, but no guarantee that the model will follow it.

# This is precisely when StructuredOutputParser comes in clutch!





