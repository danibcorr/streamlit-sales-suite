# %% Libraries

import streamlit as st
from models.inference_model import build_model, load_labels, make_prediction
from functions.language_state import StateManager

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
def load_model(model_path, labels_path):
    
    # We load the dictionary of labels
    dict_labels = load_labels(labels_path)

    # Load the model
    model = build_model(model_path = model_path, num_classes = len(dict_labels))

    return model, dict_labels

def load_image(col, language):

    if (language == 'English') or (language == 'Inglés'):
    
        label = "Upload an image from your computer."
    
    elif (language == 'Spanish') or (language == 'Español'):
    
        label = "Sube una imagen desde tu ordenador."
    
    return col.file_uploader(label, type = ['png', 'jpg', 'jpeg'])

# %% Main

if __name__ == '__main__':

    # Address of the model we are going to load.
    model_path = "models/trained_model/item_classifier"
    labels_path = "models/trained_model/lista_etiquetas.pkl"

    # We load the model
    model, labels = load_model(model_path, labels_path)

    # Create the columns
    col1, col2 = st.columns(2)

    language = state_manager.get_language()

    # Load the image we are going to process and predict its category
    file = load_image(col1, language)
    
    # Perform a prediction
    if file is not None:

        make_prediction(model, labels, file, col2)