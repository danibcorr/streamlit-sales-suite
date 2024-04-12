# %% Librerias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots

# %% Funciones

def cantidad_vendida_mes(datos, year):

    # Filtramos los datos para el año especificado
    datos_año = datos[datos['Fecha de venta'].dt.year == year]

    # Obtenemos las fechas teniendo en cuenta hasta los meses
    fechas_por_mes = datos_año['Fecha de venta'].dt.to_period('M')

    # Factorizamos para tener únicamente los valores de los meses
    eje_x = fechas_por_mes.factorize()[1]
    num_ventas_mes = np.bincount(fechas_por_mes.factorize()[0])

    # Creamos el gráfico de barras con Plotly
    fig = px.bar(datos_año, x = eje_x.astype(str), y = num_ventas_mes,
                 template = 'plotly_dark', text = num_ventas_mes)
    
    # Actualizamos los nombres de los titulos del grafico de barras
    fig.update_layout(xaxis_title = "Mes", yaxis_title = "Cantidad vendida", 
                      font=dict(size=12))

    # Mostramos los valores de cada barra encima de la barra
    fig.update_traces(texttemplate = '%{text:.2s}', textposition = 'outside')
    fig.update_layout(uniformtext_minsize = 8, uniformtext_mode = 'hide', autosize = True)

    return fig

def dinero_obtenido_mes(datos, year):

    # Filtramos los datos para el año especificado y creamos una copia explícita
    datos_año = datos[datos['Fecha de venta'].dt.year == year].copy()

    # Obtenemos las fechas teniendo en cuenta hasta los meses
    fechas_por_mes = datos_año['Fecha de venta'].dt.to_period('M')

    # Primero, necesitamos convertir la columna "Precio producto" a tipo string
    datos_año['Precio producto'] = datos_año['Precio producto'].astype(str)

    # Luego, reemplazamos las comas por puntos
    datos_año['Precio producto'] = datos_año['Precio producto'].str.replace(',', '.')

    # Finalmente, convertimos la columna "Precio producto" de nuevo a tipo float
    datos_año['Precio producto'] = datos_año['Precio producto'].astype(float)

    # Agrupamos por mes y sumamos los precios de los productos vendidos en cada mes
    dinero_obtenido_mes = datos_año.groupby(fechas_por_mes)['Precio producto'].sum()

    # Factorizamos para tener únicamente los valores de los meses
    eje_x = dinero_obtenido_mes.index.factorize()[1]
    dinero_obtenido = dinero_obtenido_mes.values

    # Creamos el gráfico de barras con Plotly
    fig = px.bar(datos_año, x = eje_x.astype(str), y = dinero_obtenido,
                 template = 'plotly_dark', text = dinero_obtenido)
    
    # Actualizamos los nombres de los titulos del grafico de barras
    fig.update_layout(xaxis_title = "Mes", yaxis_title = "Ganancias", 
                      font=dict(size=12))

    # Mostramos los valores de cada barra encima de la barra
    fig.update_traces(texttemplate = '%{text:.2s}', textposition = 'outside')
    fig.update_layout(uniformtext_minsize = 8, uniformtext_mode = 'hide', autosize = True)
    
    return fig

def categoria_productos(datos, year):

    # Filtramos los datos para el año especificado y creamos una copia explícita
    datos_año = datos[datos['Fecha de venta'].dt.year == year].copy()

    # Obtencion de las categorias
    categorias = datos_año['Tipo producto'].factorize()[1]
    num_prod_vend = np.bincount(datos_año['Tipo producto'].factorize()[0])

    # Creamos el gráfico de barras con Plotly
    fig = px.bar(datos, x = categorias.astype(str), y = num_prod_vend,
                 template = 'plotly_dark', text = num_prod_vend)
    
    # Actualizamos los nombres de los titulos del grafico de barras
    fig.update_layout(xaxis_title = "Categoría", yaxis_title = "Cantidad vendida", 
                      font=dict(size=12))

    # Mostramos los valores de cada barra encima de la barra
    fig.update_traces(texttemplate = '%{text:.2s}', textposition = 'outside')
    fig.update_layout(uniformtext_minsize = 8, uniformtext_mode = 'hide', autosize = True)

    return fig

def genero_estado(datos, year):

    # Filtramos los datos para el año especificado
    datos_año = datos[datos['Fecha de venta'].dt.year == year].copy()

    #  Agrupar segun el genero
    df_2dhist = pd.DataFrame({
        x_label: grp['Estado del producto'].value_counts()
        for x_label, grp in datos_año.groupby('Genero')
    })

    # Creamos el heatmap
    fig = px.imshow(df_2dhist, text_auto = True)

    # Actualizamos los nombres de los titulos del heatmap
    fig.update_layout(xaxis_title = "Género", yaxis_title = "Estado del producto", 
                      font=dict(size=12))

    fig.update_coloraxes(showscale=False)

    return fig

def estado_pais(datos, year):

    # Filtramos los datos para el año especificado
    datos_año = datos[datos['Fecha de venta'].dt.year == year].copy()

    # Agrupar segun el estado del producto
    df_2dhist = pd.DataFrame({
        x_label: grp['Pais'].value_counts()
        for x_label, grp in datos_año.groupby('Estado del producto')
    })

    # Creamos el heatmap
    fig = px.imshow(df_2dhist, text_auto = True)

    # Actualizamos los nombres de los titulos del heatmap
    fig.update_layout(xaxis_title = "Estado del producto", yaxis_title = "País", 
                      font=dict(size=12), height = 10)

    return fig

def data_pipeline(df, path, year):

    # Creacion de las figuras de analisis de ventas
    fig_barras = make_subplots(rows = 2, cols = 2, 
                                subplot_titles=(f"Cantidad vendida por mes en {year}",
                                                f'Dinero obtenido por mes en {year}',
                                                f'Cantidad vendida por categoria en {year}'),
                                vertical_spacing=0.15)

    # Creacion de las figuras de analisis de ventas
    fig_heatmaps = make_subplots(rows=2, cols=2,
                                specs = [[{"rowspan": 2}, {}],
                                        [None, {}]],
                                horizontal_spacing=0.15)

    # Visualización de gráficos de barras
    graph_ventas = cantidad_vendida_mes(df, year)
    graph_ganancia = dinero_obtenido_mes(df, year)
    graph_products = categoria_productos(df, year)

    # Visualización de gráficos heatmap
    graph_estado_pais = estado_pais(df, year)
    graph_genero_estado = genero_estado(df, year)
    
    # Añadimos las trazas de los gráficos de barras
    fig_barras.add_trace(graph_ventas['data'][0], row = 1, col = 1)
    fig_barras.add_trace(graph_ganancia['data'][0], row = 1, col = 2)
    fig_barras.add_trace(graph_products['data'][0], row = 2, col = 1)

    # Añadimos las trazas de los heatmaps
    fig_heatmaps.add_trace(graph_estado_pais['data'][0], row = 1, col = 1)
    fig_heatmaps.add_trace(graph_genero_estado['data'][0], row = 1, col = 2)

    # Ajustar los tamaños de las graficas de barras
    fig_barras.update_layout(height=800, width=800, 
                      title_text = f"Resultados año {year}", 
                      font=dict(size=14))

    # Mostramos las graficas de barras en streamlit
    st.plotly_chart(fig_barras) 

    # Ajustar los tamaños de las heatmaps
    fig_heatmaps.update_layout(height=800, width=800, 
                      title_text = f"Resultados año {year}", 
                      font=dict(size=14))

    fig_heatmaps.update_coloraxes(showscale=False)

    # Mostramos las graficas en streamlit
    st.plotly_chart(fig_heatmaps) 