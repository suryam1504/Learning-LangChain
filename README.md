### Learning LangChain

I am currently exploring LangChain to develop LLM-based applications, including conversational agents, retrieval-augmented generation (RAG) pipelines, and automated 
reasoning systems.

Till now, I have worked on:

1. LLMs (text completion models)

- Incorporated **gpt-3.5-turbo-instruct** model using OpenAI API key to invoke answers to some basic queries
- Understood various parameters such as temperature, max_tokens, etc.

2. Chat models

- Used **gpt-4o-mini** and noted the difference from LLMs. 
- Explored how to implement the same from other companies such as Anthropic (Claude Sonnet models) and Google (Gemini models)
- Implemented the use of open-source models from HuggingFace using both API key and installing a model on your local system.

3. Embedding models

- Worked with **text-embedding-3-small** and **text-embedding-3-large** models to convert a query and document (list of queries) into vectors (arrays with numerical 
representation of the context of the query/document) with various dimensions in the output. 
- Built a simple document similarity model which tells us which query belongs to a particular document by calculating the similarity score between the query embedding with each sample of the document embedding.

4. Prompts and Chatbot

- Static vs Dynamic Prompts (using PromptTemplate class), noted the cons of Static Prompts, and made a simple UI application, hosted on Streamlit (see 1_static_prompt.py and 2_dynamic_prompt.py)
- Created structured and reusable prompts, and chaining the template and model for efficient syntax when calling the invoke() function (see 3_prompt_generator_ui.py and 4_chain_single_invoke.py)
- Created a simple chatbot (i.e., having multi-turn conversation ability) with: 
    - i. no memory and no context (static prompt, see 5_chatbot.py)
    - ii. memory, context, and explicit mention of the User and AI roles in chat history for the LLM to follow and not get confused as conversation gets longer, both manually and using LangChain syntax (static prompt, see 6_chatbot_memory.py)
- Dynamic prompts for a list of messages using ChatPromptTemplate (dynamic prompt, see 7_chatbot_memory_dyanmic.py)
- Using Message_Placeholder to retrieve an older conversation which acts as the contextual information of a brand new conversation (see 8_Message_Placeholder)
 
5. Structured Output

Explored how to make LLMs interact with databases, APIs, and other systems using structured responses like JSON instead of unstructured text, using:

- TypedDict (defining a structured dictionary with key value pairs as the LLM output): Simple and Annotated, with Literal and Optional Arguments (see 1.2)
- Pydantic (data validation and data parsing library for python): Basic Example, Setting default values, Handling Optional Fields, Coerce, (handling implicit typecasting), EmailStr (to handle email validation), Field (default values, put contraints, write descriptions, regex, etc.) (see 2.2)
- JSON Schema (see 3.2)

6. Output Parsers

Worked with the following 4 most important output parsers out of the many which exist:

- String Output Parser: When we want string as the output from LLM
- JSON Output Parser: For JSON output, but doesn't enforce a schema and leaves things on the LLM to decide
- Structured Output Parser: Structured JSON format output which conforms to a pre-defined schema, but doesn't have data validation (eg. age should strictly be int or sentiment is strictly either pos or neg) capabilities
- Pydantic Output Parser: Structured JSON format with pre-defined schema and data validation enforced

7. Chains

Explored various types of chains to efficiently talk to LLMs:

- Simple Chain
- Sequential Chain
- Parallel Chain
- Conditional Chain

8. Runnables

Explored various types of runnables and how to connect and use them together efficiently:

- Runnable Sequence
- Runnable Parallel
- Runnable Passthrough
- Runnable Lambda
- Runnable Branch

9. Document Loaders

Explored various types of document loaders to load data from several platforms in various formats and extract their textual content, and then use LLMs to ask questions based on these documents:

- Text Loader
- PyPDFLoader
- Directory Loader
- Webbase Loader
- CSV Loader

10. Text Splitting

Explored various types of text splitters, which helps in parallelization of workflow and retaining better contextual information, and hence returns better results when passed through LLMs. Worked with following text splitters:

- Length Based
- Text Structure Based
- Document Structure Based (Eg. Python Codes and Markup languages)
- Semantic Meaning Based