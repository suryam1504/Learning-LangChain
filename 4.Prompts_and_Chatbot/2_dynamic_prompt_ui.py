from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate # for dynamic prompts only, ChatPromptTemplate is for prompt template for chat bots, see 7_chatbot_memory_dynamic.py

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini") # default is "gpt-3.5-turbo"

st.header("Research Assistant")

paper_input = st.selectbox("Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis", "The Tiger Vanishes in the Blanket"]) # this last one does print "Insufficient information available."

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Mathematical", "Code-Oriented", "Pirate"]) 

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

# template
template = PromptTemplate(
    template = """
Please summarize the research paper titled "{paper_input}" with the following specifications:  
Explanation Style: {style_input}
Explanation Length: {length_input} 
1. Mathematical Details:
    - Include relevant mathematical equations if present in the paper.
    - Explain the mathematical concepts using simple, intuitive code snippets where applicable. 
2. Analogies: 
    - Use relatable analogies to simplify complex ideas. 

If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.

Ensure the summary is clear, accurate, and aligned with the provded style and length.
""", 

input_variables=["paper_input", "style_input", "length_input"]
)

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
