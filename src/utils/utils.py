# Standard libraries
import time

# 3pps
import pandas as pd
import plotly.express as px
import streamlit as st

# List of month names
MONTHS_NAMES: list[str] = [
	"January",
	"February",
	"March",
	"April",
	"May",
	"June",
	"July",
	"August",
	"September",
	"October",
	"November",
	"December",
]


def config_streamlit_page(page_name: str) -> None:
	"""
	Configures the Streamlit page settings based on the given page name.

	Args:
		page_name: The name of the page that is being configured.
	"""

	# Configure the title of the Streamlit page
	st.title(page_name)


def check_credentials() -> bool:
	"""
	Checks whether credentials are available in the session state.

	This function verifies if the credentials are stored in the Streamlit session state.
	If available, it shows a success message for 3 seconds, and returns True.
	If not available, it shows a warning message and returns False.

	Returns:
		bool: True if credentials are available, otherwise False.
	"""

	@st.cache_data
	def credentials_available():
		# Create an empty container that can be updated
		message_container = st.empty()

		# Show a success message
		message_container.success("Credentials available.", icon="🔐")

		# Wait for 3 seconds
		time.sleep(3)

		# Empty the message
		message_container.empty()

	if "credentials" in st.session_state and st.session_state.credentials:
		credentials_available()
		return True

	st.warning("Credentials not available.", icon="⚠️")
	return False


def obtain_top(df: pd.DataFrame, top: int, column: str) -> list:
	"""Obtain the top 'n' rows from a DataFrame based on a specific column.

	Args:
		df: The DataFrame containing the data.
		top: The number of top rows to return based on the column value.
		column: The column name to sort the data by to determine the top rows.

	Returns:
		list: A list of the top 'n' values from the specified column in the DataFrame.
	"""

	return list(df[column].value_counts()[:top].index)


def summarize_year(df: pd.DataFrame, year: int) -> tuple[pd.DataFrame, dict]:
	"""
	Devuelve métricas anuales y mensuales para un año dado.

	Args:
		df: The DataFrame containing sales data.
		years: The selected years.
	"""

	df_year = df[df["Fecha de venta"].dt.year == year].copy()
	df_year["Month"] = df_year["Fecha de venta"].dt.month

	monthly = (
		df_year.groupby("Month")
		.agg(
			Revenue=("Precio producto", "sum"),
			Products_Sold=("Precio producto", "count"),
			Total_Visits=("Numero Visitas", "sum"),
			Total_Likes=("Numero de mg", "sum"),
			Avg_Duration=("Duracion de la publicacion (dias)", "mean"),
			Avg_Discount=("Descuentos (%)", "mean"),
		)
		.reindex(range(1, 13), fill_value=0)
		.reset_index()
	)

	monthly["Month_Name"] = monthly["Month"].apply(lambda m: MONTHS_NAMES[m - 1])
	monthly["Year"] = year

	annual = {
		"Year": year,
		"Total_Revenue": monthly["Revenue"].sum(),
		"Total_Products": monthly["Products_Sold"].sum(),
		"Avg_Price": df_year["Precio producto"].mean(),
		"Total_Visits": monthly["Total_Visits"].sum(),
		"Total_Likes": monthly["Total_Likes"].sum(),
		"Avg_Duration": monthly["Avg_Duration"].mean(),
		"Avg_Discount": monthly["Avg_Discount"].mean()
		if monthly["Total_Visits"].sum()
		else 0,
	}

	return monthly, annual


def plot_line(
	df: pd.DataFrame, x: str, y: str, color: str, title: str, labels: dict | None = None
) -> None:
	"""
	Plot a line chart using Plotly Express and display it in Streamlit.

	Args:
		df: The DataFrame containing the data to plot.
		x: Column name to use for the x-axis.
		y: Column name to use for the y-axis.
		color: Column name to use for grouping and coloring the lines.
		title: The chart title.
		labels: A dictionary mapping column names to
			axis/legend labels. Defaults to None.

	Returns:
		None.
	"""

	fig = px.line(df, x=x, y=y, color=color, markers=True, title=title, labels=labels)
	st.plotly_chart(fig, use_container_width=True)
