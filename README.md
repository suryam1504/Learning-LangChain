### Learning LangChain

In this repository, I am currently exploring LangChain to develop LLM-based applications, 
including conversational agents, retrieval-augmented generation (RAG) pipelines, and automated 
reasoning systems.

Till now, I have worked on:

1. LLMs (text completion models)

Incorporated **gpt-3.5-turbo-instruct** model using OpenAI API key to invoke answers to some basic queries, and understood various parameters such as temperature, 
max_tokens, etc.

2. Chat models

Used **gpt-4o-mini** and noted the difference from LLMs, explored how to implement the same from other companies such as Anthropic (Claude Sonnet models) and 
Google (Gemini models), implemented the use of open-source models from HuggingFace using both API key and installing a model on your local system.

3. Embedding models

Worked with **text-embedding-3-small** and **text-embedding-3-large** models to convert a query and document (list of queries) into vectors (arrays with numerical 
representation of the context of the query/document) with various dimensions in the output. Followed by building a simple document similarity model which tells us 
which query belongs to a particular document by calculating the similarity score between the query embedding with each sample of the document embedding.