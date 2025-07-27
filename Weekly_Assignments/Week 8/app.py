import streamlit as st
from rag_pipeline import RAGChatbot

st.set_page_config(page_title="Loan Q&A Chatbot", layout="centered")
st.title("📊 Loan Approval RAG Chatbot")

bot = RAGChatbot("data/Training Dataset.csv")

question = st.text_input("Ask a question about the loan data:")
if st.button("Get Answer") and question:
    answer = bot.get_answer(question)
    st.success(answer)