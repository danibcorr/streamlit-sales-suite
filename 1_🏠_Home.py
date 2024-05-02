# %% Libraries

import streamlit as st
from web_functions.language_state import StateManager
from deep_translator import GoogleTranslator

# %% Parameters for Streamlit

translator = GoogleTranslator(source = 'en', target = 'es')

st.set_page_config(page_title = "Home", page_icon = "🏠")
st.title("🏠 Home")

language_manager = StateManager()

# %% Functions

def select_language():

    language = language_manager.language

    text = "Translation with "
    url = "[deep-translator](https://github.com/nidhaloff/deep-translator?tab=readme-ov-file)"

    if language == 'English':
    
        language_options = ['English', 'Spanish']
        language = st.sidebar.selectbox('Select a language:', language_options)

    else:
    
        language_options = ['Inglés', 'Español']
        language = st.sidebar.selectbox('Selecciona un idioma:', language_options)
        text = translator.translate(text) 

    st.sidebar.markdown(text + " " + url)

    language_manager.language = language

    return language

def text_home():
    
    language = language_manager.language
    
    # Text related to the project summary
    title_abstract = "# Abstract"
    abstract_paragraph = """
        This project illustrates the application of **data analytics**, **data science** and **artificial intelligence** to extract meaningful information from sales data in second-hand markets. 
        sales data in second hand markets. It seeks to demonstrate competencies in these areas and to provide a practical solution for sales data management 
        for non-technical individuals. Several technologies are employed, which will be detailed in the subsequent sections.
    """

    # Text related to the data analysis section
    title_data_analysis = "# Data analysis" 
    data_analysis_paragraph = """
        The first functionality is the “**Data analysis**” page, where users can interact with graphs generated from a CSV file collected from sales data in second-tier markets. 
        data from sales data in second-hand markets. Currently, data collection is done manually and stored in Google Drive. The goal is to migrate 
        this database to a **SQL** database, which will allow real-time access and updates from this web application.
        The graphs provide information on **sales metrics**, such as monthly profit, best-selling products, and correlations between product characteristics, including gender, condition, and product 
        including gender, condition and country of sale.
    """

    # Text related to the part of the item classifier
    title_classifier = "# Item classifier"
    classifier_paragraph = """
        The second functionality is the “**Item classifier**”, which allows users to upload an image of an item for classification by an **artificial intelligence** model.
        This process seeks to automate data labeling, minimizing human intervention.
        The model, trained with **TensorFlow** and **Keras**, can identify and classify various types of items. These items can be visualized on the **Data analysis** page.
        In particular, ConvNext was used with all layers of the model frozen during training due to computational limitations (laptop with 
        RTX 3060 with 6GB RAM, i7-11800H processor with 8 cores and 16 threads, and 16GB RAM). In order to reduce memory costs, training time and inference time, we used 
        inference time, mixed-precision systems using 16-bit floating-point precision were employed. The image data, ranging from 0 to 255, were stored as 8-bit unsigned integers, 
        reducing the processing time from 5 minutes to less than 1 minute for approximately 9,000 images. To improve the learning process learning process and model convergence, 
        gradient centralization and adaptive gradient clipping were used, along with the Adam optimization algorithm with weight decay.
        Model evaluation and artifact generation were performed using MLflow.
    """

    if language == 'Spanish' or language == 'Español':

        # Translation of the content of the project summary
        title_abstract = translator.translate(title_abstract)
        abstract_paragraph = translator.translate(abstract_paragraph)

        # Translation of the content of the data analysis part
        title_data_analysis = translator.translate(title_data_analysis)
        data_analysis_paragraph = translator.translate(data_analysis_paragraph)

        # Translation of the contents of the item sorter part of the item sorter
        title_classifier = translator.translate(title_classifier)
        classifier_paragraph = translator.translate(classifier_paragraph)

    st.markdown(title_abstract)
    st.markdown(abstract_paragraph)

    st.markdown(title_data_analysis)
    st.markdown(data_analysis_paragraph)
    st.image('images/item_analysis.png')

    st.markdown(title_classifier)
    st.markdown(classifier_paragraph)
    st.image('images/item_classifier.png')

def main() -> None:
    
    select_language()
    text_home()

# %% Main

if __name__ == '__main__':

    main()