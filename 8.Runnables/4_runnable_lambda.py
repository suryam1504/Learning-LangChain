# to convert a custom python function to a runnable, and hence then this can be joined with other runnables

# simple examples

from langchain.schema.runnable import RunnableLambda

def word_counter(text):
    return len(text.split())

# converting our function to a runnable type to use its invoke method
runnable_word_counter = RunnableLambda(word_counter)

print(runnable_word_counter.invoke("Hi how are you?"))

# >>> 4


# complex example

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

parser = StrOutputParser()

passthrough = RunnablePassthrough()

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

joke_gen_chain = RunnableSequence(prompt, llm, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_counter)
})

# could also use
# parallel_chain = RunnableParallel({
#     'joke': RunnablePassthrough(),
#     'word_count': RunnableLambda(lambda x: len(x.split()))
# })

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic': 'fork'}))

# >>> {'joke': 'Why did the fork break up with the spoon? \n\nBecause it found someone who really "knows how to dish it out!"', 'word_count': 21}

# dict so keys can be extracted
