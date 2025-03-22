# more powerful then TypedDict coz better validation and hence is used more

# Note that everything inside pydantic class and object kinda goes into the prompt and hence affects the LLM output, so be as descriptive as possible

# ref from documention (https://python.langchain.com/docs/how_to/structured_output/) - Beyond just the structure of the Pydantic class, the name of the Pydantic class, the docstring, and the names and provided descriptions of parameters are very important. Most of the time with_structured_output is using a model's function/tool calling API, and you can effectively think of all of this information as being added to the model prompt.

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

# schema
class Review(BaseModel):

    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")
    

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Suryam Vivek Gupta
""")

print(result)
print("--------------")       
print(result.key_themes) # calling wrt pydantic object
print(result.cons)






# going back to pirate tone error in 1.2_with_structured_output_typeddict.py

class ExplicitReview(BaseModel):
    summary: str = Field(description="A summary of the movie, spoken like a pirate.")
    sentiment: Literal[1, 2, 3, 4, 5] = Field(description="Sentiment of the movie on a scale of 1-5, 1 being the worst, 3 being neutral, and 5 being the best.")

result1 = model.with_structured_output(ExplicitReview).invoke(
    "The movie subjectively was really enjoyable, but objectively didn't have the best flow. The acting wasn't too good, but I dont' care for that much as long as the story is interesting."
)
print(result1)

# works now!

# output
# summary="Avast, matey! Gather 'round as I spin ye a yarn 'bout a tale that be enjoyable fer yer heart, though it lacked the smooth sailin' of a fine ship! The actors, bless their souls, didnae shine like treasure, but the adventure kept me hooked like a fish on a line! So hoist the sails and enjoy the ride, even if the sea be choppy! Arrr!" sentiment=3


# another example

class ExplicitReview(BaseModel):
    summary: str = Field(description="A summary of the movie, spoken like a 3 year old kid.")
    sentiment: Literal[1, 2, 3, 4, 5] = Field(description="Sentiment of the movie on a scale of 1-5, 1 being the worst, 3 being neutral, and 5 being the best.")

result2 = model.with_structured_output(ExplicitReview).invoke(
    "The movie subjectively was really enjoyable, but objectively didn't have the best flow. The acting wasn't too good, but I dont' care for that much as long as the story is interesting."
)
print(result2)

# output
# summary='It was a fun movie with funny people and stuff happened! I liked it!' sentiment=4


# Hmm, I wonder if sentiment score gets affected by the tone of the summary, i.e. tone of a 3-year old kid might give more positive ratings vs tone of a professional critic might give more towards neutral ratings.
