import streamlit as st
from QandAbot import create_vector_db, get_qa_chain

st.title("E-learning Q and A")
btn = st.button("Create Knowledgebase")
if btn:
    pass

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer: ")
    st.write(response["result"])