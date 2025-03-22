from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')

# 1. Simple TypedDict

# schema
class Review(TypedDict):
    summary: str
    sentiment: str


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("The movie subjectively was really enjoyable, but objectively didn't have the best flow. The acting wasn't too good, but I dont' care for that much as long as the story is interesting.")

print(result)

# output
# {'summary': 'The movie was enjoyable on a personal level, despite its flaws in flow and acting. The story kept me engaged, which outweighed the shortcomings in performance.', 'sentiment': 'mixed'}

# now since this is a dictionary, we can systmatically access key value pairs

print(result['summary'])
print(result['sentiment'])


# note that nowhere in our prompt did we mention the words summary or sentiment, yet we get them because we call invoke on structured_model, which has the class Review as its input so behind the scenes a new prompt is generated which lets the LLM know that it needs to output a json with the keys summary and sentiment.

# But there could be ambiguity or confusion with just 1 or 2 words present in the key, hence we can use Annotated TypedDict.




# 2. Annotated TypedDict - Explicitly mentioning what the keys should do

#schema
class ExplicitReview(TypedDict):
    summary: Annotated[str, "A summary of the movie, spoken like a pirate."]
    sentiment: Annotated[int, "Return sentiment of the movie on a scale of 1-5, 1 being the worst, 3 being neutral, and 5 being the best."]


explicit_structured_model = model.with_structured_output(ExplicitReview)

result1 = explicit_structured_model.invoke("The movie subjectively was really enjoyable, but objectively didn't have the best flow. The acting wasn't too good, but I dont' care for that much as long as the story is interesting.")

print(result1)

# output
# {'summary': 'The movie was enjoyable on a personal level, but it had some issues with pacing and acting. Despite the flaws in performance, the engaging story kept my attention.', 'sentiment': 2}

# sentiment part is good, but it isnt returning in pirate tone, i... do not exactly know why. Setting a high temperature isn't working either.

# It seems that the LLM does not follow the instructions properly using Annotated and TypedDict, and for such things Pydantic is more suitable, see 2.2_with_structured_output_pydantic.py where it finally works.



# 3. Anyway, a more complex prompt and more things being extracted

class Review(TypedDict):
    
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive"] # Literal makes sure we get exactly the words "pos" or "neg" every time
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"] # pros is optional as they might not be present in some reviews
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]
    

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Suryam Vivek Gupta.
""")

print(result)
print(result['name']) # calling wrt dictionary format

# output
# 'name': 'Samsung Galaxy S24 Ultra Review', 'summary': 'The Samsung Galaxy S24 Ultra is an impressive flagship smartphone boasting a powerful processor, outstanding camera capabilities, and a robust battery. While it excels in performance and photography, its size and bloatware are notable downsides.', 'pros': ['Insanely powerful processor (great for gaming and productivity)', 'Stunning 200MP camera with incredible zoom capabilities', 'Long battery life with fast charging', 'S-Pen support is unique and useful'], 'cons': ['Weight and size make it uncomfortable for one-handed use', 'Bloatware includes unnecessary Samsung apps', 'High price tag of $1,300'], 'key_themes': ['Performance', 'Camera Quality', 'Battery Life', 'Design', 'Software Experience'], 'sentiment': 'pos'}

# Works great, for eg even tho cons are not mentioned explicitly in the review it still extracts it just from reading the review, but not perfect, for example it goes 'name': 'Samsung Galaxy S24 Ultra Review' instead of 'name': 'Suryam Vivek Gupta'

# Ok I ran the same code 2 more times to resample the output probabilites and it finally got my name right, but still not perfect and this method is actually considered more prone to errors as TypedDict is just a data representation and not a data validation tool.