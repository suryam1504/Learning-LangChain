# gives the same output as the input, but the llm result is atleast stored

# like in that joke example in 1_runnable_sequence

# (prompt1 --> llm --> parser) gets divided into parallel chains
# i. runnable passthrough, which just prints the joke
# ii. another runn seq (prompt2 --> llm --> parser) which expains joke

# running a simple example first

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

parser = StrOutputParser()

passthrough = RunnablePassthrough()

print(passthrough.invoke(2))
print(passthrough.invoke("hello"))
print(passthrough.invoke({'name': 'suryam'}))

# >>> 2
# hello


prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='explain the following joke - {text}',
    input_variables=['text']
)

joke_gen_chain = RunnableSequence(prompt1, llm, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, llm, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic': 'headphones'}))

# >>> {'joke': "Why did the headphone break up with its partner? \n\nBecause it couldn't handle the static in their relationship!", 'explanation': 'The joke plays on a couple of puns and double meanings. \n\n1. **Headphones and Sound:** Headphones are used to listen to audio, and "static" refers to unwanted noise or interference in sound. When headphones encounter static, it can mean that the sound quality is poor due to interference.\n\n2. **Relationship Dynamics:** The phrase "static in their relationship" is a metaphor for unresolved issues or tension that often arise in relationships, leading to problems or disagreements.\n\nSo, the humor comes from the clever wordplay: the headphone "breaks up" (stops functioning or ends a partnership) because it can\'t cope with both literal static (the audio problem) as well as the figurative static (tension and issues in a relationship). The blending of these concepts creates a light-hearted pun.'} 

# joke and explanation are in a dictionary and can be separated now