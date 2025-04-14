# to execute multiple runnables in parallel with diff types of llm

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a linkedin post about {topic}",
    input_variables=['topic']
)

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, llm,parser),
    'linkedin': RunnableSequence(prompt2, llm,parser) 
})

result = parallel_chain.invoke({'topic':'Disadvantages of using AI'})
print(result)

# >>> {'tweet': "🤖 While AI brings incredible advancements, we must be cautious of its downsides: data privacy concerns, job displacement, and bias in algorithms. Let's innovate responsibly to ensure technology benefits everyone! #AIethics #TechForGood #ResponsibleInnovation", 'linkedin': "🌐 **Navigating the AI Landscape: Understanding the Disadvantages** 🤖\n\nAs we continue to embrace the advancements of artificial intelligence across industries, it's essential to take a moment to reflect on the downsides that come along with this powerful technology. While AI boasts incredible potential, we must approach its integration with a balanced perspective. Here are some key disadvantages to consider:\n\n1. **Job Displacement**: As AI systems automate p between those who can and those who can't.\n\n6. **Complexity and Transparency**: Many AI systems operate as 'black boxes'—their decision-making processes can be opaque. This complexity makes it challenging to understand how decisions are made and can lead to trust issues among users.\n\nAs we innovate and harness the power of AI, it's crucial to address these disadvantages head-on. Let's foster conversations that prioritize ethical considerations, promote workforce development, and ensure equitable outcomes for all. \n\nWhat are your thoughts on the challenges posed by AI? Let’s discuss! 💬👇\n\n#ArtificialIntelligence #AI #Technology #Ethics #Workforce #Innovation #DataPrivacy #JobDisplacement #BiasInAI #FutureOfWork"}

# returns a dict with keys 'tweet' and 'linkedin'

print(result['tweet'])
print(result['linkedin'])