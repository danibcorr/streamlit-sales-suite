# %% Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots

# %% Functions

def quantity_sold_monthly(data, year):

    """
    Calculate and visualize the quantity sold per month for a given year.

    Args:
        data (pandas.DataFrame): The sales data.
        year (int): The year to filter the data for.

    Returns:
        plotly.graph_objs.Figure: The bar chart figure.
    """

    # Filter data for the specified year
    data_year = data[data['Fecha de venta'].dt.year == year]

    # Get dates considering up to the months
    dates_per_month = data_year['Fecha de venta'].dt.to_period('M')

    # Factorize to have only month values
    x_axis = dates_per_month.factorize()[1]
    num_sales_per_month = np.bincount(dates_per_month.factorize()[0])

    # Create bar chart with Plotly
    fig = px.bar(data_year, x=x_axis.astype(str), y=num_sales_per_month,
                 template='plotly_dark', text=num_sales_per_month)
    
    # Update title names for the bar chart
    fig.update_layout(xaxis_title="Mes", yaxis_title="Cantidad vendida", 
                      font=dict(size=12))

    # Show values of each bar on top of the bar
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', autosize=True)

    return fig

def revenue_per_month(data, year):

    """
    Calculate and visualize the revenue per month for a given year.

    Args:
        data (pandas.DataFrame): The sales data.
        year (int): The year to filter the data for.

    Returns:
        plotly.graph_objs.Figure: The bar chart figure.
    """

    # Filter data for the specified year and create an explicit copy
    data_year = data[data['Fecha de venta'].dt.year == year].copy()

    # Get dates considering up to the months
    dates_per_month = data_year['Fecha de venta'].dt.to_period('M')

    # First, we need to convert the "Product Price" column to string type
    data_year['Precio producto'] = data_year['Precio producto'].astype(str)

    # Then, replace commas with dots
    data_year['Precio producto'] = data_year['Precio producto'].str.replace(',', '.')

    # Finally, convert the "Product Price" column back to float type
    data_year['Precio producto'] = data_year['Precio producto'].astype(float)

    # Group by month and sum the prices of products sold in each month
    revenue_per_month = data_year.groupby(dates_per_month)['Precio producto'].sum()

    # Factorize to have only month values
    x_axis = revenue_per_month.index.factorize()[1]
    revenue_obtained = revenue_per_month.values

    # Create bar chart with Plotly
    fig = px.bar(data_year, x=x_axis.astype(str), y=revenue_obtained,
                 template='plotly_dark', text=revenue_obtained)
    
    # Update title names for the bar chart
    fig.update_layout(xaxis_title="Mes", yaxis_title="Ganancias", 
                      font=dict(size=12))

    # Show values of each bar on top of the bar
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', autosize=True)
    
    return fig

def product_category(data, year):

    """
    Calculate and visualize the product category distribution for a given year.

    Args:
        data (pandas.DataFrame): The sales data.
        year (int): The year to filter the data for.

    Returns:
        plotly.graph_objs.Figure: The bar chart figure.
    """

    # Filter data for the specified year and create an explicit copy
    data_year = data[data['Fecha de venta'].dt.year == year].copy()

    # Get categories
    categories = data_year['Tipo producto'].factorize()[1]
    num_products_sold = np.bincount(data_year['Tipo producto'].factorize()[0])

    # Create bar chart with Plotly
    fig = px.bar(data, x=categories.astype(str), y=num_products_sold,
                 template='plotly_dark', text=num_products_sold)
    
    # Update title names for the bar chart
    fig.update_layout(xaxis_title="Categoría", yaxis_title="Cantidad vendida", 
                      font=dict(size=12))

    # Show values of each bar on top of the bar
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', autosize=True)

    return fig

def gender_status(data, year):

    """
    Calculate and visualize the gender distribution by product status for a given year.

    Args:
        data (pandas.DataFrame): The sales data.
        year (int): The year to filter the data for.

    Returns:
        plotly.graph_objs.Figure: The heatmap figure.
    """

    # Filter data for the specified year
    data_year = data[data['Fecha de venta'].dt.year == year].copy()

    # Group by gender
    df_2dhist = pd.DataFrame({
        x_label: grp['Estado del producto'].value_counts()
        for x_label, grp in data_year.groupby('Genero')
    })

    # Create heatmap
    fig = px.imshow(df_2dhist, text_auto=True)

    # Update title names for the heatmap
    fig.update_layout(xaxis_title="Género", yaxis_title="Estado del producto", 
                      font=dict(size=12))

    fig.update_coloraxes(showscale=False)

    return fig

def status_country(data, year):

    """
    Calculate and visualize the product status distribution by country for a given year.

    Args:
        data (pandas.DataFrame): The sales data.
        year (int): The year to filter the data for.

    Returns:
        plotly.graph_objs.Figure: The heatmap figure.
    """

    # Filter data for the specified year
    data_year = data[data['Fecha de venta'].dt.year == year].copy()

    # Group by product status
    df_2dhist = pd.DataFrame({
        x_label: grp['Pais'].value_counts()
        for x_label, grp in data_year.groupby('Estado del producto')
    })

    # Create heatmap
    fig = px.imshow(df_2dhist, text_auto=True)

    # Update title names for the heatmap
    fig.update_layout(xaxis_title="Estado del producto", yaxis_title="País", 
                      font=dict(size=12), height=10)

    return fig

def data_pipeline(df, path, year):

    """
    Create sales analysis figures for a given year.

    Args:
        df (pandas.DataFrame): The sales data.
        path (str): The path to save the figures.
        year (int): The year to filter the data for.
    """
    
    # Create sales analysis figures
    fig_bars = make_subplots(rows=2, cols=2, 
                                subplot_titles=(f"Cantidad vendida por mes en {year}",
                                                f'Dinero obtenido por mes en {year}',
                                                f'Cantidad vendida por categoria en {year}'),
                                vertical_spacing=0.15)

    # Create heatmap analysis figures
    fig_heatmaps = make_subplots(rows=2, cols=2,
                                specs=[[{"rowspan": 2}, {}],
                                        [None, {}]],
                                horizontal_spacing=0.15)

    # Visualize bar charts
    graph_sales = quantity_sold_monthly(df, year)
    graph_revenue = revenue_per_month(df, year)
    graph_categories = product_category(df, year)

    # Visualize heatmaps
    graph_status_country = status_country(df, year)
    graph_gender_status = gender_status(df, year)
    
    # Add traces of bar charts
    fig_bars.add_trace(graph_sales['data'][0], row=1, col=1)
    fig_bars.add_trace(graph_revenue['data'][0], row=1, col=2)
    fig_bars.add_trace(graph_categories['data'][0], row=2, col=1)

    # Add traces of heatmaps
    fig_heatmaps.add_trace(graph_status_country['data'][0], row=1, col=1)
    fig_heatmaps.add_trace(graph_gender_status['data'][0], row=1, col=2)

    # Adjust sizes of bar charts
    fig_bars.update_layout(height=800, width=800, 
                      title_text = f"Resultados año {year}", 
                      font=dict(size=14))

    # Show bar charts in Streamlit
    st.plotly_chart(fig_bars) 

    # Adjust sizes of heatmaps
    fig_heatmaps.update_layout(height=800, width=800, 
                      title_text = f"Resultados año {year}", 
                      font=dict(size=14))

    fig_heatmaps.update_coloraxes(showscale=False)

    # Show heatmaps in Streamlit
    st.plotly_chart(fig_heatmaps)