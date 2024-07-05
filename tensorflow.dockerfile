# Tomamos la imagen de Tensorflow
FROM tensorflow/tensorflow:2.16.2-gpu

# Directorio de trabajo
WORKDIR /home/project

# Copiamos los archivos de la carpeta del proyecto del Host al Contenedor
COPY project /home/project

# Instalamos los requerimentos
RUN pip install --upgrade pip && \
    pip install --ignore-installed -r requirements.txt

# Expose the port to be used by the application
EXPOSE 8501

# Command to start the application
CMD ["streamlit", "run", "1_🏠_Home.py"]