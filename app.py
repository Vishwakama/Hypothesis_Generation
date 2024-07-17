import streamlit as st
from langchain.llms import CTransformers
from transformers import pipeline

# Function to generate questions based on the hypothesis
def generate_questions(hypothesis):
    # Initialize the question generation pipeline
    #pip install -U langchain-community
    qg_pipeline = pipeline("question-generation")

    # Generate questions based on the hypothesis
    questions = qg_pipeline(hypothesis)
    return questions

# Function to get response from LLAMA 2 model
def get_llama_response(input_text):
    # LLAMA 2 model
    llm = CTransformers(model='model\llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})

    # Generate the response from the LLAMA 2 model
    response = llm(input_text)
    return response

# Streamlit UI
st.set_page_config(page_title="Generate Questions", page_icon='❓', layout='centered', initial_sidebar_state='collapsed')
st.title("Generate Questions ❓")

# Input text
input_text = st.text_area("Enter Text:", height=200)

# Generate hypothesis button
if st.button("Generate Hypothesis"):
    if input_text:
        hypothesis = get_llama_response(input_text)
        st.write("Generated Hypothesis:")
        st.write(hypothesis)
    else:
        st.write("Please enter some text to generate a hypothesis.")

# Generate questions button
if st.button("Generate Questions"):
    if hypothesis:
        questions = generate_questions(hypothesis)
        st.write("Generated Questions:")
        for i, q in enumerate(questions, start=1):
            st.write(f"{i}. {q['question']}")
    else:
        st.write("Please generate a hypothesis first.")
