# Standard libraries
import random
from datetime import UTC, datetime, timedelta

# 3pps
import pandas as pd
import streamlit as st

# Own modules
from config import MONTHS_NAMES, REQUIRED_COLUMNS
from config.constants import (
	COL_AVG_DISCOUNT,
	COL_DESCUENTOS,
	COL_FECHA_VENTA,
	COL_MONTH,
	COL_MONTH_NAME,
	COL_PRECIO_PRODUCTO,
	COL_PRODUCTS_SOLD,
	COL_REVENUE,
	COL_YEAR,
	SESSION_CREDENTIALS,
)


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

	if (
		SESSION_CREDENTIALS in st.session_state
		and st.session_state[SESSION_CREDENTIALS]
	):
		if not st.session_state.get("_credentials_shown"):
			st.toast("Credentials available.", icon="🔐")
			st.session_state._credentials_shown = True
		return True

	st.warning("Credentials not available.", icon="⚠️")
	return False


def obtain_top(df: pd.DataFrame, top: int, column: str) -> list[str]:
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

	yearly_sales = df[df[COL_FECHA_VENTA].dt.year == year].copy()
	yearly_sales[COL_MONTH] = yearly_sales[COL_FECHA_VENTA].dt.month

	monthly_summary = (
		yearly_sales.groupby(COL_MONTH)
		.agg(
			Revenue=(COL_PRECIO_PRODUCTO, "sum"),
			Products_Sold=(COL_PRECIO_PRODUCTO, "count"),
			Avg_Discount=(COL_DESCUENTOS, "mean"),
		)
		.reindex(range(1, 13), fill_value=0)
		.reset_index()
	)

	monthly_summary[COL_MONTH_NAME] = monthly_summary[COL_MONTH].apply(
		lambda m: MONTHS_NAMES[m - 1]
	)
	monthly_summary[COL_YEAR] = year

	annual_summary = {
		COL_YEAR: year,
		"Total_Revenue": monthly_summary[COL_REVENUE].sum(),
		"Total_Products": monthly_summary[COL_PRODUCTS_SOLD].sum(),
		"Avg_Price": yearly_sales[COL_PRECIO_PRODUCTO].mean(),
		COL_AVG_DISCOUNT: monthly_summary[COL_AVG_DISCOUNT].mean(),
	}

	return monthly_summary, annual_summary


def generate_synthetic_data(num_samples: int = 1000) -> pd.DataFrame:
	"""
	Generate synthetic sales data for testing.

	Args:
		num_samples: Number of rows to generate.

	Returns:
		DataFrame with columns matching REQUIRED_COLUMNS schema.
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
		fecha = (
			datetime.now(tz=UTC) - timedelta(days=random.randint(0, 365))  # nosec
		).strftime(  # nosec
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
	df[COL_FECHA_VENTA] = pd.to_datetime(df[COL_FECHA_VENTA])

	return df
