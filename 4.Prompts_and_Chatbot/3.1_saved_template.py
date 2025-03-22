from langchain_core.prompts import PromptTemplate # for dynamic prompts only, ChatPromptTemplate is for prompt template for chat bots, see 4_chatbot_ui.py

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

# save the template to a JSON file
template.save("4.Prompts_and_Chatbot/3.1_saved_template.json")