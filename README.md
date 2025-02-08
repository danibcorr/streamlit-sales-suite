# 📊 Streamline Sales Suite

## **📄 Overview**

Streamline Sales Suite is a comprehensive platform designed for data analysis and
visualization.

## **🚀 Getting Started**

To set up and use this repository, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   ```

2. **Install Poetry (if not already installed):**

   ```bash
   pip install poetry
   ```

3. **Install Dependencies:** Set up the Python virtual environment and install all
   necessary dependencies:

   ```bash
   poetry install
   ```

4. **Acquire Training Images:** Obtain the images required to train the classification
   model:

   ```bash
   poetry run python ./src/data_acquisition.py
   ```

5. **Train the Classification Model:** Train the CNN model with the acquired images:

   ```bash
   poetry run python ./src/training_model.py
   ```

6. **Run the Application:** Launch the complete project using Streamlit:

   ```bash
   poetry run streamlit run ./src/1_🏠_Home.py
   ```

7. **Docker Deployment:** A Dockerfile and Docker Compose are included for containerizing
   the application, which is particularly useful for deployment after the model is
   trained.

**Note:** The data analysis component relies on a private dataset and may not be
functional without it. However, the project can be adapted to work with other datasets.

## **🌟 Contributing**

Contributions are highly encouraged! Whether you have new tools, models, or techniques to
share, your input is welcome. Please feel free to submit a pull request or open an issue
to discuss your ideas.

## **🤖 License**

This project is licensed under the MIT License, allowing you to freely use, modify, and
distribute the code.
