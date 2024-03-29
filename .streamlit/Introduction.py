import streamlit as st

st.set_page_config(
    page_title='Introduction',
    page_icon='👋',
)

st.write('# Welcome to the Paper Recommender! 👋')

st.markdown(
"""
    This application will take in your preferences among categories and return the most relevant recent arXiv papers

    # How to use

    1. Click into the Recommender tab
    2. Choose your preferences in the 'Preferences' menu
    4. Click 'Run'!

"""
)