**📊 Streamline Sales Suite**
=====================

**📄 Summary**
-----------

This project is a comprehensive data analysis and visualization platform using Python, Pandas, NumPy, and deep learning techniques. It includes a convolutional neural network (CNN) model for image classification, utilizing computer vision techniques to obtain classes from different items.

**🗂️ Directory structure**
-------------------------

The project has a well-organized directory structure with various folders and files associated with data analysis, modeling, and web functions.

+ **data**
  * `data_analysis.py`: A script for data analysis, generating necessary plots associated with sales data.
  * `data_scraping.py`: A script for scraping images from Bing using web scraping techniques, which can be cleaned and processed later with `data_cleaning.py`.

+ **models**
  * `layers/`: A folder containing functions that implement different optimizers, learning rate schedulers, and similar.
  * `architecture.py`: A script defining the base model architecture.
  * `config.py`: A script containing parameters for model training configuration and MLflow artifact creation.
  * `inference_model.py`: A script for using the trained model for inference in Streamlit.

+ **pages**
  * `2_📊_Data_Analysis.py`: A page for visualizing and interacting with data graphs.
  * `3_🔎_Item_Classifier.py`: A page for performing item classification.

+ **web_functions**
  * `language_State.py`: A script that keeps track of the selected language for page translation.

**🚀 Using this repo**
---------------------

To use this repository, follow these steps:

1. **Install required libraries**: Run `pip install -r requirements.txt` to install the necessary libraries.
2. **Scrape images**: Run `python data_scraping.py` to obtain images for training the classification model.
3. **Clean and process data**: Use `data_cleaning.py` to clean and process the scraped images.
4. **Create dataset structure**: Create a `datasets` folder in the project root, with subfolders named after the image labels. For example:
    ```
    VENTASYADIRA
    │
    ├── datasets
    │   ├── books
    │   │   ├── file_1.png
    │   │   └──...
    │   ├── cloth
    │   │   ├── file_1.png
    │   │   └──...
    ```
5. **Train the model**: Run `python training_model.py` to train the CNN model.
6. **Run the Streamlit app**: Run `streamlit run 1_🏠_Home.py` to execute the Streamlit app locally and use the article classifier.

**Note**: The data analysis part will not be functional, as it depends on the private dataset used in this project. However, it can be adapted to other datasets.

**🌟 Contributions**
-------------------

Contributions are welcome! If you have new tools, models, or techniques you'd like to share, I'd be delighted.

**🤖 License**
-------------

This project is distributed under the MIT License. Feel free to play, modify, and share the code as you wish!