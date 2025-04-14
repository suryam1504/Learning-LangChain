# to conditionally route input to diff chains or runnables based on custom logic

# example where we generate a reoprt on a topic, if it is less thn 500 great, o/w we ask llm to summarize it again

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

parser = StrOutputParser()

passthrough = RunnablePassthrough()

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text:\n {text}',
    input_variables=['text']
)

report_gen_chain = RunnableSequence(prompt1, llm, parser)

# branch_chain = RunnableBranch(
#     (condition1, runnable which gets executed if condition is true),
#     (condition, runnable which gets executed if condition is true),
#     ....
#     (default)
# )

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, llm, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({"topic":'Python vs C++'}))











# another example could be where we use an llm to categorize an email into complaint, refund, or general query, and accordingly return an output