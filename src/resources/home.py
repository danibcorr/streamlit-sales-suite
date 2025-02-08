import streamlit as st

from src.utils import config_streamlit_page


def home_text():

    # Configure the columns
    col1, col2 = st.columns(2, gap="large")

    # Insert the content for the first column
    with col1:
        st.image(
            image="./imgs/nikola-duza-fi6kmznklGQ-unsplash.jpg", use_container_width=True
        )

    # Insert the content for the second column
    with col2:
        st.header("📊 Streamlit Sales Suite")
        st.markdown(
            """
            This project was created with the aim of simplifying the process of analyzing
            product sales data for second-hand markets. Although the data is not publicly
            available, the code is licensed under the MIT License. This tool currently
            supports two main features: the ability to visualize predefined charts for
            data analysis, as well as a tool for direct interaction with the data.
            """
        )

        st.subheader("Charts")
        st.markdown(
            """
            This feature allows you to visualize charts based on the collected data.
            """
        )


# First call to the config page function
config_streamlit_page(page_name="🏠 Home")

# Display the text
home_text()
