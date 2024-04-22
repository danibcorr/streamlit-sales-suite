# %% Libraries

import streamlit as st
from models.inference_model import load_model, load_labels, make_prediction
from web_functions.language_state import StateManager

# %% Definitions for streamlit

if 'language' not in st.session_state:
    
    st.session_state.language = 'English'

state_manager = StateManager(language=st.session_state.language)

language = state_manager.get_language()

if (language == 'English') or (language == 'Inglés'):

    st.set_page_config(page_title = "Item Classifier", page_icon = "🔎")
    st.title("🔎 Item Classifier")

elif (language == 'Spanish') or (language == 'Español'):

    st.set_page_config(page_title = "Clasificación de artículos", page_icon = "🔎")
    st.title("🔎 Clasificación de artículos")
    
# %% Functions

@st.cache_resource
def load_model_inference(model_path: str, labels_path: str) -> tuple:

    """
    Load the model and labels from the given paths.

    Args:
        model_path (str): The path to the model file.
        labels_path (str): The path to the labels file.

    Returns:
        tuple: A tuple containing the loaded model and labels.
    """

    dict_labels = load_labels(labels_path)
    model = load_model(model_path = model_path, num_classes = len(dict_labels))

    return model, dict_labels

def load_image(col, language: str) -> st.columns:

    """
    Create a file uploader for the given column and language.

    Args:
        col: The Streamlit column to create the uploader in.
        language (str): The language to use for the uploader label.

    Returns:
        st.uploaded_file_manager: The file uploader object.
    """

    if (language == 'English') or (language == 'Inglés'):

        label = "Upload an image from your computer."

    elif (language == 'Spanish') or (language == 'Español'):

        label = "Sube una imagen desde tu ordenador."

    return col.file_uploader(label, type = ['png', 'jpg', 'jpeg'])

def main(model_path: str, labels_path: str) -> None:
    
    """
    Main function to load the model and perform image classification.

    Args:
        model_path (str): The path to the model file.
        labels_path (str): The path to the labels file.
    """

    model, labels = load_model_inference(model_path, labels_path)
    col1, col2 = st.columns(2)
    language = state_manager.get_language()
    file = load_image(col1, language)

    if file is not None:

        make_prediction(model, labels, file, col2)

# %% Main

if __name__ == '__main__':

    model_path = "models/trained_model/item_classifier"
    labels_path = "models/trained_model/dict_labels.pkl"
    main(model_path, labels_path)