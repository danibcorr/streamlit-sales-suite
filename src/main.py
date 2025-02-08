import json

import streamlit as st
import pandas as pd


def upload_credentials() -> None:
    """
    Handles the uploading of a JSON file containing credentials and stores them
    in the session state. It also loads a corresponding Excel file based on the
    credentials and stores the resulting DataFrame in the session state.
    """

    # Title displayed to upload credentials
    st.sidebar.title("Load credentials")

    # Input to upload a JSON file
    uploaded_file = st.sidebar.file_uploader(
        "Upload the JSON file with credentials", type=["json"]
    )

    # Button to save credentials
    if uploaded_file is not None:
        try:
            # Read JSON file
            credentials = json.load(uploaded_file)

            # Save in session_state the value of credentials
            st.session_state.credentials = credentials
            st.sidebar.success("Credentials successfully uploaded", icon="✅")

            # Attempt to load the Excel file if path is valid
            if "path" in credentials:
                st.session_state.dataframe = pd.read_excel(credentials["path"])
            else:
                st.sidebar.error("Invalid credentials: Missing path key.", icon="⚠️")
        except Exception as e:
            st.sidebar.error(f"Error reading the JSON file: {e}", icon="⚠️")


def streamlit_configuration() -> None:
    """
    Configures the Streamlit app with page settings, project resources, and credentials
    handling.
    """

    # Page configuration for the project
    st.set_page_config(
        page_title="Streamlit Sales Suite",
        page_icon=":material/insert_chart:",
        layout="wide",
    )

    # Pages used for the project
    pages = {
        "Home": [
            st.Page("./resources/home.py", title="Home", icon=":material/home:"),
        ],
        "Resources": [
            st.Page(
                "./resources/graphs.py",
                title="Graphs",
                icon=":material/monitoring:",
            )
        ],
    }

    # Logo for the project
    st.logo(
        image="./imgs/logo.png",
        link="https://streamlit.io/gallery",
        size="large",
    )

    # Initialize credentials and dataframe in session_state with default values
    st.session_state.setdefault("credentials", None)
    st.session_state.setdefault("dataframe", None)

    # Button for uploading credentials for access to the data
    upload_credentials()

    # Execution of the page configuration
    pg = st.navigation(pages)
    pg.run()


# Load the streamlit app configuration
streamlit_configuration()
