import streamlit as st


class StateManager:

    def __init__(self):
        """
        Constructs all the necessary attributes for the StateManager object.
        If 'language' is not in st.session_state, English will be the default language.
        """

        if "language" not in st.session_state:

            # English will be the default language
            st.session_state.language = "English"

    @property
    def language(self):
        """
        Gets the current language of the Streamlit application.

        Returns:
            str: A string representing the language of the Streamlit application.
        """

        return st.session_state.language

    @language.setter
    def language(self, value):
        """
        Sets the language of the Streamlit application.

        Args:
            value (str): A string representing the language to be set in the Streamlit application.
        """

        st.session_state.language = value
