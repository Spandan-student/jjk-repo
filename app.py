import streamlit as st

user_input = st.text_input("Write the line")

st.subheader(user_input)