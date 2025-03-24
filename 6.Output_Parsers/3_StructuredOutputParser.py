from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema # note that unlike others this is not in langchain_core

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
    ResponseSchema(name='fact_4', description='Fact 4 about the topic'),
    ResponseSchema(name='fact_5', description='Fact 5 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.format(topic='Walmart')
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)

chain = template | model | parser

result = chain.invoke({'topic':'Walmart'})

print(result)

# {'fact_1': 'Walmart was founded by Sam Walton in 1962 in Rogers, Arkansas.', 'fact_2': "It is the world's largest retailer, operating over 10,500 stores in 24 countries.", 'fact_3': "Walmart employs approximately 2.3 million people globally, making it one of the world's largest employers.", 'fact_4': 'The company has a strong commitment to sustainability, aiming to be powered by 100% renewable energy by 2035.', 'fact_5': "Walmart's services include groceries, pharmaceuticals, and financial services, aiming to be a one-stop shop for customers."}



# Great! But this also has a con, which is we still cannot have data validation. For example, if we want to extract name, age, and gender of a celebrity, we cannot enforce that the age should be a number (int). LLM could very well return age as "35 years," which might be a problem in certain scenarios.

# Aaaaand this we exactly where Pydantic comes in clutch!
