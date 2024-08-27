# We take the image from Tensorflow
FROM tensorflow/tensorflow:2.16.2-gpu

# Work directory
WORKDIR /home/project

# Copy dependency files first to leverage Docker cache
COPY ./pyproject.toml ./poetry.lock ./

# Install the required dependencies
RUN pip install --no-cache-dir poetry \
    && poetry install --no-root \
    && rm -rf /root/.cache/pip

# Copy the rest of the application files
COPY ./src ./src

# Expose the port to be used by the application
EXPOSE 8501

# Command to start the application
CMD ["poetry", "run", "streamlit", "run", "./src/1_🏠_Home.py"]