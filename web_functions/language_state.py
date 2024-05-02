# %% Libraries

import streamlit as st

# %% Class definition

class StateManager:

    def __init__(self):

        if 'language' not in st.session_state:

            # English will be the default language
            st.session_state.language = 'English'  

    @property
    def language(self):

        return st.session_state.language

    @language.setter
    def language(self, value):

        st.session_state.language = value