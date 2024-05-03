# %% Libraries

import streamlit as st
import pandas as pd
from models.inference_model import load_model, load_labels, make_prediction
from web_functions.language_state import StateManager
from deep_translator import GoogleTranslator
import datetime

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
    
    # Load labels from the given labels path
    dict_labels = load_labels(labels_path)
    
    # Load the model from the given model path with the number of classes equal to the number of labels
    model = load_model(model_path = model_path, num_classes = len(dict_labels))
    
    # Return the loaded model and labels as a tuple
    return model, dict_labels

def load_image(col) -> st.columns:

    """
    Create a file uploader for the given column.

    Args:
        col: The Streamlit column to create the uploader in.

    Returns:
        st.uploaded_file_manager: The file uploader object.
    """
    
    # Define the label for the file uploader
    label = "Upload an image from your computer."
    
    # Translate the label if the language is not English
    if language!= 'English':

        label = translator.translate(label)
    
    # Create a file uploader in the given column with the specified label and allowed file types
    return col.file_uploader(label, type = ['png', 'jpg', 'jpeg'])

def data_filling(dataset_path: str, predictions_string: str) -> None:

    """
    Fill in missing data in the dataset based on the predictions and user input.

    Args:
        dataset_path (str): The path to the dataset CSV file.
        predictions_string (str): The predicted product type.

    Returns:
        None
    """
    
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(dataset_path)
    
    # Filter the data to obtain statistics based on the predicted product type
    df_product = df[df['Tipo producto'] == predictions_string]
    
    # Convert "Precio producto" to float once, replacing commas with dots
    df_product["Precio producto"] = df_product["Precio producto"].str.replace(',', '.').astype('float32')
    
    # Precompute modes and mean
    modes = df_product[["Plataforma de venta", "Pais", "Genero", "Estado del producto"]].mode().iloc[0]
    mean_price = df_product["Precio producto"].mean()
    
    # Get the column names
    column_names = df.columns.tolist()
    
    # Create a form with input fields for each column
    with st.form("my_form"):

        inputs = {}
        
        # Iterate over each column
        for column in column_names:

            # Handle different column types
            if column in modes.index:

                # Use mode value for categorical columns
                inputs[column] = st.text_input(column, value =modes[column])

            elif column == "Fecha de venta":

                # Use current date for date column
                now = datetime.datetime.now()
                inputs[column] = st.text_input(column, value = f"{str(now.day).zfill(2)}/{str(now.month).zfill(2)}/{now.year}")
           
            elif column == "Precio producto":

                # Use mean price for price column
                inputs[column] = st.text_input(column, value = mean_price)

            elif column == "Tipo producto":

                # Use predicted product type for product type column
                inputs[column] = st.text_input(column, value = predictions_string)

            else:

                # Use empty input for other columns
                inputs[column] = st.text_input(column)
        
        # Add a submit button to the form
        submitted = st.form_submit_button("OK")
    
    # If the form is submitted, add the new row to the dataframe
    if submitted:

        # Create a new row from the input values
        new_row = {k: [v] for k, v in inputs.items()}
        new_row = pd.DataFrame(new_row)

        # Concatenate the new row with the original dataframe
        df = pd.concat([df, new_row], ignore_index = True)

        # Save the updated dataframe to the CSV file
        df.to_csv(dataset_path, index = False)
    
    # Display the updated dataframe
    st.dataframe(df, use_container_width = True)

def main(model_path: str, labels_path: str, dataset_path: str) -> None:

    """
    Main function to load the model, perform image classification, and fill in missing data.

    Args:
        model_path (str): The path to the model file.
        labels_path (str): The path to the labels file.
        dataset_path (str): The path to the dataset CSV file.
    """
    
    # Load the model and labels
    model, labels = load_model_inference(model_path, labels_path)
    
    # Create two columns in the Streamlit app
    col1, col2 = st.columns(2)
    
    # Create a file uploader in the first column
    file = load_image(col1)
    
    # If a file is uploaded, make a prediction and fill in missing data
    if file is not None:

        # Make a prediction using the uploaded image
        predictions_string = make_prediction(model, labels, file, col2)
        
        # Fill in missing data in the dataset based on the prediction
        data_filling(dataset_path, predictions_string)

# %% Main

if __name__ == '__main__':

    dataset_path = "datasets/Datos Ventas.csv"
    model_path = "models/trained_model/item_classifier"
    labels_path = "models/trained_model/dict_labels.pkl"

    main(model_path, labels_path, dataset_path)