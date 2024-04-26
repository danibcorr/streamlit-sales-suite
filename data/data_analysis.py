# %% Libraries

import numpy as np
import pandas as pd
import streamlit as st
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from web_functions.language_state import StateManager

# %% Translations

def translation(year: int, year1: int = 0, year2: int = 0) -> dict:

    translations = {
        'English': {
            'fig_bars_titles': (f"Quantity sold per month in {year}", None, f'Quantity sold by category in {year}',
                                None, f'Money made per month in {year}'),
            'fig_heatmaps_titles': (f'State/country ratio {year}', f'Gender/state ratio {year}',
                                    None, f'Money earned per month in {year}'),
            'fig_bars_first_title': f"Results year {year}",
            'title_quantity': f"Quantity sold per month in {year1} and {year2}",
            'title_revenue': f"Amount of money earned per month in {year1} and {year2}",
            'title_categories': f"Amount sold per category in {year1} and {year2}",
            'title_heatmap_1': [f"State/country ratio in {year1}", f"State/country ratio in {year2}"],
            'title_heatmap_2': [f"Gender/State Ratio in {year1}", f"Gender/State Ratio in {year2}"],
            'xaxis_title': "Month",
            'xaxis_title_2': "Category",
            'yaxis_title': "Quantity sold",
            'yaxis_title_2': "Profit",
            'legend_title': "Year"
        },
        'Spanish': {
            'fig_bars_titles': (f"Cantidad vendida por mes en {year}", None, f'Cantidad vendida por categoria en {year}',
                                None, f'Dinero obtenido por mes en {year}'),
            'fig_heatmaps_titles': (f"Relación estado/país {year}", f'Relación género/estado {year}',
                                    None, f'Dinero obtenido por mes en {year}'),
            'fig_bars_first_title': f"Resultados año {year}",
            'title_quantity': f"Cantidad vendida por mes en {year1} y {year2}",
            'title_revenue': f"Dinero obtenido por mes en {year1} y {year2}",
            'title_categories': f"Cantidad vendida por categoría en {year1} y {year2}",
            'title_heatmap_1': [f"Relación estado/país en {year1}", f"Relación estado/país en {year2}"],
            'title_heatmap_2': [f"Relación género/estado en {year1}", f"Relación género/estado en {year2}"],
            'xaxis_title': "Mes",
            'xaxis_title_2': "Categoria",
            'yaxis_title': "Cantidad vendida",
            'yaxis_title_2': "Ganancias",
            'legend_title': "Año"
        }
    }

    return translations

# %% Globals

COLOR_BAR_1 = 'rgb(255, 224, 189)'
COLOR_BAR_2 = 'rgb(175, 238, 238)'

# %% Definitions for streamlit

if 'language' not in st.session_state:
    
    st.session_state.language = 'English'

state_manager = StateManager(language=st.session_state.language)
language = state_manager.get_language()
lang_key = 'English' if language == 'English' or language == 'Inglés' else 'Spanish'

# %% Functions

def quantity_sold_monthly(data: pd.DataFrame, year: int) -> plotly.graph_objects.Figure:

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
    
    # Show values of each bar on top of the bar
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', autosize=True)

    return fig

def revenue_per_month(data: pd.DataFrame, year: int) -> plotly.graph_objects.Figure:

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
    
    # Show values of each bar on top of the bar
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', autosize=True)
    
    return fig

def product_category(data: pd.DataFrame, year: int) -> plotly.graph_objects.Figure:

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

    # Show values of each bar on top of the bar
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', autosize=True)

    return fig

def gender_status(data: pd.DataFrame, year: int) -> plotly.graph_objects.Figure:

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
    return px.imshow(df_2dhist, text_auto=True)

def status_country(data: pd.DataFrame, year: int) -> plotly.graph_objects.Figure:

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
    return px.imshow(df_2dhist, text_auto=True)

def data_pipeline(df: pd.DataFrame, path: str, year: int) -> None:
    
    """
    Generate sales analysis and heatmap figures for a given year.
    This function generates two types of figures: bar charts and heatmaps.
    The bar charts show sales quantity and revenue by month, as well as product categories.
    The heatmaps show relationships between state/country and gender/state.
    The figures are displayed in Streamlit using `st.plotly_chart`.

    Parameters:
    df (pd.DataFrame): Input data frame containing sales data.
    path (str): Path to save the figures (not used in this implementation).
    year (int): Year for which to generate the analysis figures.

    Returns:
    None
    """

    # Obtain the translations
    translations = translation(year)
    print(lang_key)

    # Create sales analysis figures
    fig_bars = make_subplots(rows = 2, cols = 4, 
                            subplot_titles = translations[lang_key]['fig_bars_titles'],
                            specs = [[{"colspan": 2}, {}, {"colspan":2,"rowspan":2}, {}], [{"colspan": 2}, {}, {}, {}]])

    # Create heatmap analysis figures
    fig_heatmaps = make_subplots(rows = 2, cols = 3,
                                subplot_titles = translations[lang_key]['fig_heatmaps_titles'],
                                specs = [[{"rowspan": 2}, {}, None], [{}, None, None]],
                                horizontal_spacing = 0.15)

    # Visualize bar charts
    graph_sales = quantity_sold_monthly(df, year)
    graph_revenue = revenue_per_month(df, year)
    graph_categories = product_category(df, year)

    # Visualize heatmaps
    graph_status_country = status_country(df, year)
    graph_gender_status = gender_status(df, year)
    
    # Add traces of bar charts
    fig_bars.add_trace(graph_sales['data'][0], row=1, col=1)
    fig_bars.add_trace(graph_revenue['data'][0], row=2, col=1)
    fig_bars.add_trace(graph_categories['data'][0], row=1, col=3)

    # Add traces of heatmaps
    fig_heatmaps.add_trace(graph_status_country['data'][0], row=1, col=1)
    fig_heatmaps.add_trace(graph_gender_status['data'][0], row=1, col=2)

    # Adjust sizes of bar charts
    fig_bars.update_layout(height=800,
                           # Set width to None to adapt to Streamlit page width
                           width = None,  
                           title_text = translations[lang_key]['fig_bars_first_title'],
                           # Center the title
                           title_x = 0.5,  
                           font = dict(size = 14))

    # Show bar charts in Streamlit
    # Set use_container_width to True to adapt to Streamlit page width
    st.plotly_chart(fig_bars, use_container_width=True)  

    # Adjust sizes of heatmaps
    fig_heatmaps.update_layout(height = 800, 
                               width = 800)

    fig_heatmaps.update_coloraxes(showscale = False)

    # Show heatmaps in Streamlit
    st.plotly_chart(fig_heatmaps, use_container_width=True)  

def compare_years(data: pd.DataFrame, year1: int, year2: int) -> None:
    
    """
    Compare sales data between two years.
    This function generates overlapped bar charts and heatmaps to compare sales data between two years.
    The bar charts show sales quantity, revenue, and product categories for each year.
    The heatmaps show relationships between state/country and gender/state for each year.
    The figures are displayed in Streamlit using `st.plotly_chart`.

    Parameters:
    data (pd.DataFrame): Input data frame containing sales data.
    year1 (int): First year to compare.
    year2 (int): Second year to compare.

    Returns:
    None
    """

    # Obtain the translations
    translations = translation(year = None, year1 = year1, year2 = year2)

    # Filter data for each year
    data_year1 = data[data['Fecha de venta'].dt.year == year1]
    data_year2 = data[data['Fecha de venta'].dt.year == year2]

    # Create figures for each year
    fig_year1_quantity = quantity_sold_monthly(data_year1, year1)
    fig_year2_quantity = quantity_sold_monthly(data_year2, year2)

    fig_year1_revenue = revenue_per_month(data_year1, year1)
    fig_year2_revenue = revenue_per_month(data_year2, year2)

    fig_year1_categories = product_category(data_year1, year1)
    fig_year2_categories = product_category(data_year2, year2)

    # Create overlapped figures
    fig_overlapped_quantity = go.Figure()
    fig_overlapped_quantity.add_bar(name = f"{year1}", x = fig_year1_quantity['data'][0]['x'], y = fig_year1_quantity['data'][0]['y'], marker_color = COLOR_BAR_1)
    fig_overlapped_quantity.add_bar(name = f"{year2}", x = fig_year2_quantity['data'][0]['x'], y = fig_year2_quantity['data'][0]['y'], marker_color = COLOR_BAR_2)

    fig_overlapped_revenue = go.Figure()
    fig_overlapped_revenue.add_bar(name = f"{year1}", x = fig_year1_revenue['data'][0]['x'], y = fig_year1_revenue['data'][0]['y'], marker_color = COLOR_BAR_1)
    fig_overlapped_revenue.add_bar(name = f"{year2}", x = fig_year2_revenue['data'][0]['x'], y = fig_year2_revenue['data'][0]['y'], marker_color = COLOR_BAR_2)

    fig_overlapped_categories = go.Figure()
    fig_overlapped_categories.add_bar(name = f"{year1}", x = fig_year1_categories['data'][0]['x'], y = fig_year1_categories['data'][0]['y'], marker_color = COLOR_BAR_1)
    fig_overlapped_categories.add_bar(name = f"{year2}", x = fig_year2_categories['data'][0]['x'], y = fig_year2_categories['data'][0]['y'], marker_color = COLOR_BAR_2)

    # Add titles
    fig_overlapped_quantity.update_layout(title_text = translations[lang_key]['title_quantity'], xaxis_title = translations[lang_key]['xaxis_title'], 
                                            yaxis_title = translations[lang_key]['yaxis_title'], legend_title = translations[lang_key]['legend_title'])

    fig_overlapped_revenue.update_layout(title_text = translations[lang_key]['title_revenue'], xaxis_title = translations[lang_key]['xaxis_title'], 
                                            yaxis_title = translations[lang_key]['yaxis_title_2'], legend_title = translations[lang_key]['legend_title'])

    fig_overlapped_categories.update_layout(title_text = translations[lang_key]['title_categories'], xaxis_title = translations[lang_key]['xaxis_title_2'], 
                                            yaxis_title = translations[lang_key]['yaxis_title'], legend_title = translations[lang_key]['legend_title'])

    # Show overlapped figures in Streamlit
    st.plotly_chart(fig_overlapped_quantity, use_container_width = True)
    st.plotly_chart(fig_overlapped_revenue, use_container_width = True)
    st.plotly_chart(fig_overlapped_categories, use_container_width = True)

    # Create heatmap analysis figures with increased spacing
    fig_heatmaps_1 = make_subplots(rows = 2, cols = 2,
                                   subplot_titles = translations[lang_key]['title_heatmap_1'],
                                   specs = [[{"rowspan": 2}, {"rowspan": 2}], [{}, {}]],
                                   vertical_spacing = 0.25,
                                   horizontal_spacing = 0.25)

    fig_heatmaps_2 = make_subplots(rows = 1, cols = 2,
                                   subplot_titles = translations[lang_key]['title_heatmap_2'],
                                   specs = [[{}, {}]],
                                   vertical_spacing = 0.25,
                                   horizontal_spacing = 0.25)

    # Visualize heatmaps
    fig_year1_gender_status = gender_status(data_year1, year1)
    fig_year2_gender_status = gender_status(data_year2, year2)
    fig_year1_status_country = status_country(data_year1, year1)
    fig_year2_status_country = status_country(data_year2, year2)

    # Add traces of heatmaps
    fig_heatmaps_1.add_trace(fig_year1_status_country['data'][0], row = 1, col = 1)
    fig_heatmaps_1.add_trace(fig_year2_status_country['data'][0], row = 1, col = 2)
    fig_heatmaps_2.add_trace(fig_year1_gender_status['data'][0], row = 1, col = 1)
    fig_heatmaps_2.add_trace(fig_year2_gender_status['data'][0], row = 1, col = 2)

    # Adjust sizes and color settings for heatmaps
    fig_heatmaps_1.update_layout(height = 800, width = 800)
    fig_heatmaps_1.update_coloraxes(showscale = False)
    fig_heatmaps_2.update_layout(height = 400, width = 800)
    fig_heatmaps_2.update_coloraxes(showscale = False)

    # Show heatmaps in Streamlit
    st.plotly_chart(fig_heatmaps_1)
    st.plotly_chart(fig_heatmaps_2)