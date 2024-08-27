import streamlit as st
from deep_translator import GoogleTranslator
from web_functions.language_state import StateManager

# Initialize the Google Translator with English as the source language and Spanish as the target language
translator = GoogleTranslator(source="en", target="es")

# Configure the Streamlit page
st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")
st.title("🏠 Home")

# Manage language state
language_manager = StateManager()


def select_language() -> str:
    """
    Allow the user to select the language for the app and update the language state.

    Returns:
        str: The selected language.
    """
    language = language_manager.language

    # Set text for language selection info
    text = "Translation with "
    url = "[deep-translator](https://github.com/nidhaloff/deep-translator?tab=readme-ov-file)"

    # Define language options based on the current language
    if language == "English":
        language_options = ["English", "Spanish"]
        language = st.sidebar.selectbox("Select a language:", language_options)
    else:
        language_options = ["Inglés", "Español"]
        language = st.sidebar.selectbox("Selecciona un idioma:", language_options)
        text = translator.translate(text)

    # Display translation info
    st.sidebar.markdown(text + url)

    # Update the language state
    language_manager.language = language

    return language


def display_home_content() -> None:
    """
    Display the content of the home page.
    """
    col1, col2 = st.columns(2)
    language = language_manager.language

    # Define the text content for different sections
    content = {
        "abstract": {
            "title": "# Abstract",
            "text": """
                This project uses **data analytics**, **data science**, and **AI** to generate insights from second-hand market sales data. 
                It demonstrates proficiency in these areas while providing a practical tool for non-technical users to manage sales data. 
                More details on the technologies used are provided below.
            """,
        },
        "data_analysis": {
            "title": "# Data Analysis",
            "text": """
                The "**Data Analysis**" page lets users interact with sales data visualizations from a CSV file. Currently, data collection is manual and stored in Google Drive, 
                but the goal is to move this to a **SQL** database for real-time updates. The visualizations include **sales metrics** like monthly profit, top-selling products, 
                and correlations between product attributes (e.g., gender, condition, country).
            """,
        },
        "classifier": {
            "title": "# Item Classifier",
            "text_1": """
                The "**Item Classifier**" allows users to upload an item image for classification by an **AI** model, automating data labeling and reducing manual effort. 
                The model, trained with **TensorFlow** and **Keras**, classifies items, which are then visualized on the **Data Analysis** page.
            """,
            "text_2": """
                We used ConvNext with frozen layers during training due to hardware limitations (RTX 3060, 6GB RAM, i7-11800H). To reduce memory and time, 
                mixed-precision systems with 16-bit floating-point precision were employed, speeding up image processing from 5 minutes to under 1 minute for 9,000 images.
            """,
            "text_3": """
                For improved learning and model performance, gradient centralization and adaptive gradient clipping were applied, along with the Adam optimizer with weight decay. 
                Model evaluation and artifact generation were handled using MLflow.
            """,
        },
    }

    # Translate the content if needed
    if language in ["Spanish", "Español"]:
        for section in content.values():
            section["title"] = translator.translate(section["title"])
            section["text"] = translator.translate(section["text"])

    # Display the content
    with col1:
        st.markdown(content["abstract"]["title"])
        st.markdown(content["abstract"]["text"])
        st.markdown(content["data_analysis"]["title"])
        st.markdown(content["data_analysis"]["text"])
        st.image("src/images/item_analysis.png")

    with col2:
        st.markdown(content["classifier"]["title"])
        st.markdown(content["classifier"]["text_1"])
        st.markdown(content["classifier"]["text_2"])
        st.markdown(content["classifier"]["text_3"])
        st.image("src/images/item_classifier.png")


def main() -> None:
    """
    The main function to run the application.
    """
    select_language()
    display_home_content()


if __name__ == "__main__":
    main()
