import streamlit as st
from utils.query import pg_query
from numpy import dot

st.set_page_config(layout="wide")

## helper function to convert the text format of the scores to floats
def convert_to_list_of_floats(s):
    ## remove the curly braces and split the string by comma
    elements = s.strip('{}').split(',')
    ## convert each element to float and return as list
    return [float(e) for e in elements]

## initialize postgres for queries
pg = pg_query()

## query for daily and weekly score tables
daily_scores = pg.query(
    """
    select *
    from public.scores
    where updated::timestamp > now() - interval '1 day'
    """
)

weekly_scores = pg.query(
    """
    select *
    from public.scores
    """
)

## give column names to the data frames
daily_scores.columns = ['id', 'str_scores', 'title', 'updated']
weekly_scores.columns = ['id', 'str_scores', 'title', 'updated']

## convert scores to numeric data type using helper
daily_scores['scores'] = daily_scores['str_scores'].apply(convert_to_list_of_floats)
weekly_scores['scores'] = weekly_scores['str_scores'].apply(convert_to_list_of_floats)

## giant menu for the user to change the preferences if needed
with st.expander('Preferences', expanded=False):
    fma_input = st.number_input('Foundational Models and Architectures', value=0.4)
    lug_input = st.number_input('Language Understanding and Generation', value=0.4)
    tla_input = st.number_input('Transfer Learning and Adaptation', value=0.5)
    mclm_input = st.number_input('Multilingual and Cross-Lingual Models', value=0.7)
    ebf_input = st.number_input('Ethics, Bias, and Fairness', value=0.0)
    ie_input = st.number_input('Interpretability and Explainability', value=0.3)
    eb_input = st.number_input('Evaluation and Benchmarks', value=0.0)
    ire_input = st.number_input('Information Retrieval and Extraction', value=0.7)
    aonl_input = st.number_input('Applications of NLP and LLMs', value=0.7)
    remd_input = st.number_input('Resource-Efficient Models and Deployment', value=0.5)
    ttp_input = st.number_input('Tokenization and Text Processing', value=1.0)

## run button
run = st.button('Run')

## actions taken when run button is hit
if run:
    ## set the chosen preferences to be a vector
    preference = [fma_input, lug_input, tla_input, mclm_input, ebf_input, ie_input, eb_input, ire_input, aonl_input, remd_input, ttp_input]

    ## perform dot product between preference vector and score vectors
    daily_scores['preference'] = daily_scores['scores'].map(lambda x: dot(x, preference))
    weekly_scores['preference'] = weekly_scores['scores'].map(lambda x: dot(x, preference))
    
    ## grab the highest scoring paper from preferences based on time scale
    daily_max = daily_scores.loc[daily_scores['preference'].idxmax()]
    weekly_max = weekly_scores.loc[weekly_scores['preference'].idxmax()]

    ## output the chosen papers with links
    st.write('Daily Most Relevant Article:')
    st.write('[', daily_max['title'], '](', daily_max['id'], ')')

    st.text('')
    st.text('')

    st.write('Weekly Most Relevant Article:')
    st.write('[', weekly_max['title'], '](', weekly_max['id'], ')')

    ## expandable menu in case the user wishes to see the full tables and their scores
    with st.expander('Full Tables', expanded=False):
        st.write(daily_scores[['id', 'title', 'scores', 'preference']].sort_values(by='preference', ascending=False))
        st.write(weekly_scores[['id', 'title', 'scores', 'preference']].sort_values(by='preference', ascending=False))
