# %% Libraries

import streamlit as st
import pandas as pd
from models.inference_model import load_model, load_labels, make_prediction
from web_functions.language_state import StateManager
from deep_translator import GoogleTranslator

# %% Definitions for streamlit

translator = GoogleTranslator(source = 'en', target = 'es')
language_manager = StateManager()
language = language_manager.language

page_title = "Item classifier"
page_title = translator.translate(page_title) if language != 'English' else page_title
page_icon = "🔎"

st.set_page_config(page_title = page_title, page_icon = page_icon, layout = "wide")
st.title(f"{page_icon} {page_title}")

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

def load_image(col) -> st.columns:

    """
    Create a file uploader for the given column.

    Args:
        col: The Streamlit column to create the uploader in.

    Returns:
        st.uploaded_file_manager: The file uploader object.
    """

    label = "Upload an image from your computer."
    label = label if language == 'English' else translator.translate(label)

    return col.file_uploader(label, type = ['png', 'jpg', 'jpeg'])

def data_filling(dataset_path: str, predictions_string: str) -> None:
    
    # Read the CSV file
    df = pd.read_csv(dataset_path)
    
    # Get the column names
    column_names = df.columns.tolist()

    # Create a form with input fields for each column
    with st.form("my_form"):

        inputs = {}
        
        for column in column_names:

            if column == "Tipo producto":
            
                inputs[column] = st.text_input(column, value = predictions_string)
            
            else:
            
                inputs[column] = st.text_input(column)
        
        submitted = st.form_submit_button("OK")

    # If the form is submitted, add the new row to the dataframe
    if submitted:
    
        new_row = {k: [v] for k, v in inputs.items()}
        new_row = pd.DataFrame(new_row)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(dataset_path, index=False)

    st.dataframe(df, use_container_width=True)

def main(model_path: str, labels_path: str, dataset_path: str) -> None:
    
    """
    Main function to load the model and perform image classification.

    Args:
        model_path (str): The path to the model file.
        labels_path (str): The path to the labels file.
    """

    model, labels = load_model_inference(model_path, labels_path)
    col1, col2 = st.columns(2)
    file = load_image(col1)

    if file is not None:

        predictions_string = make_prediction(model, labels, file, col2)
        data_filling(dataset_path, predictions_string)

# %% Main

if __name__ == '__main__':

    dataset_path = "./datasets/Datos Ventas.csv"
    model_path = "models/trained_model/item_classifier"
    labels_path = "models/trained_model/dict_labels.pkl"

    main(model_path, labels_path, dataset_path)