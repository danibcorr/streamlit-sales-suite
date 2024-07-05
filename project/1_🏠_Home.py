# %% Libraries

import streamlit as st
from web_functions.language_state import StateManager
from deep_translator import GoogleTranslator

# %% Parameters for Streamlit

# Create a Google Translator instance with English as the source language and Spanish as the target language
translator = GoogleTranslator(source = 'en', target = 'es')

# Set the page configuration for Streamlit
st.set_page_config(page_title = "Home", page_icon = "🏠")
st.title("🏠 Home")

# Create a StateManager instance to manage the language state
language_manager = StateManager()

# %% Functions

def select_language() -> str:

    """
    Select the language for the application.

    Returns:
        str: The selected language.
    """

    language = language_manager.language

    # Set the text and URL for the language selection
    text = "Translation with "
    url = "[deep-translator](https://github.com/nidhaloff/deep-translator?tab=readme-ov-file)"

    # Depending on the current language, set the language options and translate the text
    if language == 'English':

        language_options = ['English', 'Spanish']
        language = st.sidebar.selectbox('Select a language:', language_options)

    else:

        language_options = ['Inglés', 'Español']
        language = st.sidebar.selectbox('Selecciona un idioma:', language_options)
        text = translator.translate(text)

    # Display the language selection text and URL
    st.sidebar.markdown(text + " " + url)

    # Update the language state
    language_manager.language = language

    return language

def text_home() -> None:

    """
    Display the home page content.
    """

    language = language_manager.language

    # Text related to the project summary
    title_abstract = "# Abstract"
    abstract_paragraph = """
        This project applies **data analytics**, **data science**, and **artificial intelligence** to derive valuable insights 
        from sales data in second-hand markets. The aim is to showcase proficiency in these fields while offering a practical 
        solution for managing sales data, particularly for non-technical individuals. The technologies utilized will be 
        elaborated in the following sections.
    """

    # Text related to the data analysis section
    title_data_analysis = "# Data analysis" 
    data_analysis_paragraph = """
        The initial feature is the "**Data Analysis**" page, which enables users to interact with graphs generated from a CSV 
        file of sales data from second-hand markets. At present, data collection is performed manually and housed in Google Drive. 
        The objective is to transition this database to a **SQL** database, facilitating real-time access and updates via this web 
        application. The graphs offer insights into **sales metrics**, such as monthly profit, top-selling products, and correlations 
        among product attributes, including gender, condition, and country of sale.
    """

    # Text related to the part of the item classifier
    title_classifier = "# Item classifier"
    classifier_paragraph = """
        The second feature is the "**Item Classifier**", which enables users to upload an image of an item for classification by an 
        **artificial intelligence** model. This process aims to automate data labeling, thereby reducing human intervention. 
        The model, trained using **TensorFlow** and **Keras**, can identify and classify various types of items, which can then be 
        visualized on the **Data Analysis** page.
    """
    classifier_paragraph_2 = """
        Specifically, ConvNext was utilized with all model layers frozen during training due to computational constraints 
        (laptop with RTX 3060 with 6GB RAM, i7-11800H processor with 8 cores and 16 threads, and 16GB RAM). To decrease memory costs, 
        training time, and inference time, mixed-precision systems employing 16-bit floating-point precision were used. Image data, 
        ranging from 0 to 255, were stored as 8-bit unsigned integers, reducing the processing time from 5 minutes to less than 1 minute 
        for approximately 9,000 images.
    """
    classifier_paragraph_3 = """
        To enhance the learning process and model convergence, gradient centralization and adaptive gradient clipping were implemented, 
        along with the Adam optimization algorithm with weight decay. Model evaluation and artifact generation were conducted using MLflow.
    """

    # Translate the content if the language is not English
    if language == 'Spanish' or language == 'Español':

        title_abstract = translator.translate(title_abstract)
        abstract_paragraph = translator.translate(abstract_paragraph)
        title_data_analysis = translator.translate(title_data_analysis)
        data_analysis_paragraph = translator.translate(data_analysis_paragraph)
        title_classifier = translator.translate(title_classifier)
        classifier_paragraph = translator.translate(classifier_paragraph)
        classifier_paragraph_2 = translator.translate(classifier_paragraph_2)
        classifier_paragraph_3 = translator.translate(classifier_paragraph_3)

    # Display the content
    st.markdown(title_abstract)
    st.markdown(abstract_paragraph)

    st.markdown(title_data_analysis)
    st.markdown(data_analysis_paragraph)
    st.image('images/item_analysis.png')

    st.markdown(title_classifier)
    st.markdown(classifier_paragraph)
    st.markdown(classifier_paragraph_2)
    st.markdown(classifier_paragraph_3)
    st.image('images/item_classifier.png')

def main() -> None:

    """
    Main entry point of the application.
    """

    select_language()
    text_home()

# %% Main

if __name__ == '__main__':

    main()