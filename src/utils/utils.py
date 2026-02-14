# Standard libraries
import random
import time
from datetime import datetime, timedelta

# 3pps
import pandas as pd
import plotly.express as px
import streamlit as st

# Own modules
from config import MONTHS_NAMES, REQUIRED_COLUMNS


def config_streamlit_page(page_name: str) -> None:
	"""
	Sets the displayed title of the current Streamlit page.

	Args:
		page_name: The title to display at the top of the page.

	Returns:
		None.
	"""

	st.title(page_name)


def check_credentials() -> bool:
	"""
	Verifies whether credentials exist in the Streamlit
	session state and displays a temporary status message
	accordingly.

	Returns:
		Whether valid credentials are present in the
			session state.
	"""

	@st.cache_data
	def credentials_available() -> None:
		status_message = st.empty()
		status_message.success("Credentials available.", icon="🔐")
		time.sleep(3)
		status_message.empty()

	if "credentials" in st.session_state and st.session_state.credentials:
		credentials_available()
		return True

	st.warning("Credentials not available.", icon="⚠️")
	return False


def obtain_top(df: pd.DataFrame, top: int, column: str) -> list:
	"""
	Returns the most frequently occurring values in a
	DataFrame column, ranked by descending frequency.

	Args:
		df: The DataFrame containing the data.
		top: The number of most frequent values to
			return.
		column: The column name to compute value
			frequencies from.

	Returns:
		The most frequent values from the specified
			column.
	"""

	return list(df[column].value_counts().iloc[:top].index)


def summarize_year(df: pd.DataFrame, year: int) -> tuple[pd.DataFrame, dict]:
	"""
	Computes monthly and annual sales metrics for a given
	year. The monthly DataFrame includes revenue, products
	sold, average discount, and month names for all 12
	months (zero-filled where no data exists). The annual
	dictionary aggregates totals and averages.

	Args:
		df: The DataFrame containing sales data.
		year: The year to summarize.

	Returns:
		A tuple of the monthly breakdown DataFrame and
			the annual summary dictionary.
	"""

	yearly_sales = df[df["Fecha de venta"].dt.year == year].copy()
	yearly_sales["Month"] = yearly_sales["Fecha de venta"].dt.month

	monthly_summary = (
		yearly_sales.groupby("Month")
		.agg(
			Revenue=("Precio producto", "sum"),
			Products_Sold=("Precio producto", "count"),
			Avg_Discount=("Descuentos (%)", "mean"),
		)
		.reindex(range(1, 13), fill_value=0)
		.reset_index()
	)

	monthly_summary["Month_Name"] = monthly_summary["Month"].apply(
		lambda m: MONTHS_NAMES[m - 1]
	)
	monthly_summary["Year"] = year

	annual_summary = {
		"Year": year,
		"Total_Revenue": monthly_summary["Revenue"].sum(),
		"Total_Products": monthly_summary["Products_Sold"].sum(),
		"Avg_Price": yearly_sales["Precio producto"].mean(),
		"Avg_Discount": monthly_summary["Avg_Discount"].mean(),
	}

	return monthly_summary, annual_summary


def plot_line(
	df: pd.DataFrame, x: str, y: str, color: str, title: str, labels: dict | None = None
) -> None:
	"""
	Renders an interactive Plotly line chart with markers
	inside a full-width Streamlit container.

	Args:
		df: The DataFrame containing the data to plot.
		x: Column name to use for the x-axis.
		y: Column name to use for the y-axis.
		color: Column name to use for grouping and
			coloring the lines.
		title: The chart title.
		labels: A dictionary mapping column names to
			axis/legend labels.

	Returns:
		None.
	"""

	fig = px.line(df, x=x, y=y, color=color, markers=True, title=title, labels=labels)
	st.plotly_chart(fig, width="stretch")


def generate_synthetic_data(num_samples: int = 1000) -> pd.DataFrame:
	"""_summary_

	Args:
		num_samples (int, optional): _description_. Defaults to 1000.

	Returns:
		pl.DataFrame: _description_
	"""

	plataformas = ["Amazon", "Shopify", "Etsy", "eBay", "Mercado Libre"]
	paises = ["España", "México", "Argentina", "Colombia", "EE.UU."]
	generos = ["M", "F"]
	tipos = ["Electrónica", "Ropa", "Hogar", "Libros", "Deportes"]
	estados = ["Nuevo", "Usado - Como nuevo", "Usado - Buen estado"]

	data = []

	for _ in range(num_samples):
		tipo = random.choice(tipos)  # nosec
		precio = (
			round(random.uniform(10, 500), 2)  # nosec
			if tipo != "Libros"
			else round(random.uniform(5, 50), 2)  # nosec
		)
		fecha = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime(  # nosec
			"%Y-%m-%d"
		)

		data.append(
			[
				random.choice(plataformas),  # nosec
				fecha,
				random.choice(paises),  # nosec
				random.choice(generos),  # nosec
				precio,
				tipo,
				random.choice(estados),  # nosec
				random.choice([0, 5, 10, 15, 20, 50]),  # nosec
			]
		)

	df = pd.DataFrame(data, columns=REQUIRED_COLUMNS)
	df["Fecha de venta"] = pd.to_datetime(df["Fecha de venta"])

	return df
