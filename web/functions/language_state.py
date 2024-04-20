# %% Libraries

import streamlit as st

# %% Class definition

class StateManager:

    def __init__(self, language):

        self.language = language
        st.session_state.language = self.language

    def get_language(self):

        return st.session_state.language

    def set_language(self, language):

        st.session_state.language = language