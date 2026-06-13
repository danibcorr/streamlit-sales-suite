# Standard libraries
import random
import time
from datetime import UTC, datetime, timedelta

# 3pps
import polars as pl
import streamlit as st

# Own modules
from config import MONTHS_NAMES, REQUIRED_COLUMNS
from config.constants import (
	COL_AVG_DISCOUNT,
	COL_AVG_PRICE,
	COL_DESCUENTO_APLICADO,
	COL_FECHA_VENTA,
	COL_MONTH,
	COL_MONTH_NAME,
	COL_PRECIO_PRODUCTO,
	COL_PRODUCTS_SOLD,
	COL_REVENUE,
	COL_TOTAL_PRODUCTS,
	COL_TOTAL_REVENUE,
	COL_YEAR,
	SESSION_CREDENTIALS,
	SESSION_DATAFRAME,
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

	@st.cache_data
	def _show_credentials_status() -> None:
		status_message = st.empty()
		status_message.success("Credentials available.", icon="🔐")
		time.sleep(3)
		status_message.empty()

	if (
		SESSION_CREDENTIALS in st.session_state
		and st.session_state[SESSION_CREDENTIALS]
		and st.session_state.get(SESSION_DATAFRAME) is not None
	):
		_show_credentials_status()
		return True

	st.warning("Credentials not available.", icon="⚠️")
	return False


def obtain_top(df: pl.DataFrame, top: int, column: str) -> list[str]:
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

	return (
		df.group_by(column)
		.len()
		.sort("len", descending=True)
		.head(top)
		.get_column(column)
		.to_list()
	)


def filter_by_year(df: pl.DataFrame, year: int) -> pl.DataFrame:
	"""
	Returns the DataFrame filtered to the given year.

	Args:
		df: The DataFrame containing sales data with a date column.
		year: The year to filter by.

	Returns:
		A filtered DataFrame.
	"""

	return df.filter(pl.col(COL_FECHA_VENTA).dt.year() == year)


def summarize_year(df: pl.DataFrame, year: int) -> tuple[pl.DataFrame, dict]:
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

	yearly_sales = filter_by_year(df, year)

	monthly_agg = (
		yearly_sales.with_columns(pl.col(COL_FECHA_VENTA).dt.month().alias(COL_MONTH))
		.group_by(COL_MONTH)
		.agg(
			pl.col(COL_PRECIO_PRODUCTO).sum().alias(COL_REVENUE),
			pl.col(COL_PRECIO_PRODUCTO).count().alias(COL_PRODUCTS_SOLD),
			pl.col(COL_DESCUENTO_APLICADO).mean().alias(COL_AVG_DISCOUNT),
		)
	)

	all_months = pl.DataFrame({COL_MONTH: list(range(1, 13))}).cast(
		{COL_MONTH: monthly_agg.schema[COL_MONTH]}
	)

	monthly_summary = (
		all_months.join(monthly_agg, on=COL_MONTH, how="left")
		.fill_null(0)
		.sort(COL_MONTH)
		.with_columns(
			pl.col(COL_MONTH)
			.map_elements(lambda m: MONTHS_NAMES[m - 1], return_dtype=pl.Utf8)
			.alias(COL_MONTH_NAME),
			pl.lit(year).alias(COL_YEAR),
		)
	)

	annual_summary = {
		COL_YEAR: year,
		COL_TOTAL_REVENUE: monthly_summary.get_column(COL_REVENUE).sum(),
		COL_TOTAL_PRODUCTS: monthly_summary.get_column(COL_PRODUCTS_SOLD).sum(),
		COL_AVG_PRICE: (
			yearly_sales.get_column(COL_PRECIO_PRODUCTO).mean()
			if yearly_sales.height > 0
			else 0.0
		),
		COL_AVG_DISCOUNT: monthly_summary.get_column(COL_AVG_DISCOUNT).mean(),
	}

	return monthly_summary, annual_summary


def generate_synthetic_data(num_samples: int = 1000) -> pl.DataFrame:
	"""
	Generates synthetic sales data matching the required columns.

	Args:
		num_samples: Number of rows to generate. Defaults to 1000.

	Returns:
		A Polars DataFrame with synthetic sales data.
	"""

	platforms = ["Amazon", "Shopify", "Etsy", "eBay", "Mercado Libre"]
	countries = ["España", "México", "Argentina", "Colombia", "EE.UU."]
	genders = ["M", "F"]
	product_types = ["Electrónica", "Ropa", "Hogar", "Libros", "Deportes"]
	statuses = ["Nuevo", "Usado - Como nuevo", "Usado - Buen estado"]

	data = []

	for _ in range(num_samples):
		product_type = random.choice(product_types)  # nosec
		price = (
			round(random.uniform(10, 500), 2)  # nosec
			if product_type != "Libros"
			else round(random.uniform(5, 50), 2)  # nosec
		)
		sale_date = (
			datetime.now(tz=UTC) - timedelta(days=random.randint(0, 365))  # nosec
		).strftime("%Y-%m-%d")

		data.append(
			{
				REQUIRED_COLUMNS[0]: random.choice(platforms),  # nosec
				REQUIRED_COLUMNS[1]: sale_date,
				REQUIRED_COLUMNS[2]: random.choice(countries),  # nosec
				REQUIRED_COLUMNS[3]: random.choice(genders),  # nosec
				REQUIRED_COLUMNS[4]: round(random.uniform(1, 100), 2),  # nosec
				REQUIRED_COLUMNS[5]: random.choice([0, 5, 10, 15, 20, 50]),  # nosec
				REQUIRED_COLUMNS[6]: random.choice([0, 5, 10, 15, 20]),  # nosec
				REQUIRED_COLUMNS[7]: price,
				REQUIRED_COLUMNS[8]: product_type,
				REQUIRED_COLUMNS[9]: random.choice(statuses),  # nosec
				REQUIRED_COLUMNS[10]: random.randint(0, 500),  # nosec
				REQUIRED_COLUMNS[11]: random.randint(0, 10000),  # nosec
			}
		)

	df = pl.DataFrame(data).with_columns(
		pl.col(COL_FECHA_VENTA).str.to_date("%Y-%m-%d").alias(COL_FECHA_VENTA)
	)

	return df
