# we can define schema and data validation now  

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

class Person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    gender: Literal["Male", "Female", "Other"] = Field(description='Gender of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age, gender and city of a {country} celebrity. \n {format_instruction}',
    input_variables=['country'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({'country':'African'})

print(final_result)
print(type(final_result))

# Indian
# name='Priyanka Chopra' age=41 gender='Female' city='Mumbai'
# <class '__main__.Person'>

# USA
# name='Taylor Swift' age=33 gender='Female' city='New York'  # oh man gpt-4o-mini loves taylot swift

# Australian
# name='Margot Robbie' age=33 gender='Female' city='Sydney'

# African
# name="Lupita Nyong'o" age=40 gender='Female' city='Nairobi'


# the actual prompt which goes into LLM

print(template.invoke({'country':'African'}))

# text='Generate the name, age, gender and city of a African celebrity. \n The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {"name": {"description": "Name of the person", "title": "Name", "type": "string"}, "age": {"description": "Age of the person", "exclusiveMinimum": 18, "title": "Age", "type": "integer"}, "gender": {"description": "Gender of the person", "enum": ["Male", "Female", "Other"], "title": "Gender", "type": "string"}, "city": {"description": "Name of the city the person belongs to", "title": "City", "type": "string"}}, "required": ["name", "age", "gender", "city"]}\n```'

# The parser adds so much specific info!