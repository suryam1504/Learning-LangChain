# How to define multiple prompts separately and use their reference in the main file so the code doesnt' look bulky and its efficient and reuable

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini") # default is "gpt-3.5-turbo"

st.header("Research Assistant")

paper_input = st.selectbox("Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis", "The Tiger Vanishes in the Blanket"]) # this last one does print "Insufficient information available."

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Mathematical", "Code-Oriented", "Pirate"]) 

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

# template
# no need to define a huge template here, it's in 3.1_saved_template.py now which was converted to a JSON format

# load the template from the JSON file
template = load_prompt("4.Prompts_and_Chatbot/3.1_saved_template.json")
# Similarly we can save and use multiple templates in the same file

# fill the placeholders
prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})


if st.button("Summarize"):
    result = model.invoke(prompt)
    st.write(result.content)

# notice how we use invoke() function 2 times here, once for the prompt template and once for the model 
