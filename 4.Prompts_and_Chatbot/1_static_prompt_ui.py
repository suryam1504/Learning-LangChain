from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini") # default is "gpt-3.5-turbo"

st.header("Research Assistant")

user_input = st.text_input("Enter your query here")

if st.button("Summarize"):
    result = model.invoke(user_input)
    st.write(result.content)

# Static prompts not recommended as it gives the user too much power to influence the input query which can greatly affect the output from the LLM, hence dynamic prompts are recommended.
