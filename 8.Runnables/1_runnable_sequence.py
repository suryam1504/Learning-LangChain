# to connect multiple runnable components (like prompttemplates, llms, etc.) sequentially
# in these, output from 1 is input to the next one in chain

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

parser = StrOutputParser()

# prompt = PromptTemplate(
#     template="Write a joke about {topic}",
#     input_variables=['topic']
# )

# chain = RunnableSequence(prompt, llm, parser)

# print(chain.invoke({'topic':'AI'}))

# >>> Why did the AI go to therapy?

# It couldn't stop overthinking everything!



# more lenghty chains

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(prompt1, llm, parser, prompt2, llm, parser)

print(chain.invoke({'topic':'math'}))

# >>> The joke "Why was the equal sign so humble? Because it knew it wasn’t less than or greater than anyone else!" plays on the mathematical properties of the equal sign (=) and the concepts of inequality (less than < and greater than >).

# In mathematics, the equal sign signifies a relationship where two values are the same. It is "humble" because it doesn't assert superiority or inferiority — it simply states that both sides of the equation are equal. The humor lies in anthropomorphizing the equal sign, attributing to it a personality trait (humility) based on its function in math.