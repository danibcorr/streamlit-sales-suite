# Standard libraries
from datetime import date
from unittest.mock import MagicMock, patch

# 3pps
import polars as pl
import pytest

# Own modules
from config.constants import (
	COL_DESCUENTO_APLICADO,
	COL_ESTADO_PRODUCTO,
	COL_FECHA_VENTA,
	COL_GENERO,
	COL_PAIS,
	COL_PRECIO_PRODUCTO,
	COL_TIPO_PRODUCTO,
)
from resources.graphs import (
	category_sales_by_period,
	compare_income_month_years,
	compare_products_years,
	flow_money,
	gender_country,
	gender_status,
	money_month,
	product_month,
	status_country,
)


class TestMoneyMonth:
	"""
	Tests for the money_month function.
	"""

	@pytest.fixture
	def sales_df(self) -> pl.DataFrame:
		"""
		Provides a DataFrame with sales data for testing.

		Returns:
			A DataFrame with sale dates and product prices.
		"""

		return pl.DataFrame(
			{
				COL_FECHA_VENTA: [
					date(2024, 1, 15),
					date(2024, 1, 20),
					date(2024, 3, 10),
					date(2023, 6, 1),
				],
				COL_PRECIO_PRODUCTO: [100.0, 200.0, 150.0, 50.0],
			}
		)

	@patch("resources.graphs.st")
	def test_money_month_renders(
		self, mock_st: MagicMock, sales_df: pl.DataFrame
	) -> None:
		"""
		Verifies that money_month calls Streamlit functions.

		Returns:
			None.
		"""

		money_month(sales_df, 2024)
		mock_st.subheader.assert_called_once()
		mock_st.plotly_chart.assert_called_once()


class TestProductMonth:
	"""
	Tests for the product_month function.
	"""

	@pytest.fixture
	def sales_df(self) -> pl.DataFrame:
		"""
		Provides a DataFrame with sales data for testing.

		Returns:
			A DataFrame with sale dates, product types, and prices.
		"""

		return pl.DataFrame(
			{
				COL_FECHA_VENTA: [
					date(2024, 1, 15),
					date(2024, 1, 20),
					date(2024, 3, 10),
				],
				COL_TIPO_PRODUCTO: ["A", "B", "A"],
				COL_PRECIO_PRODUCTO: [100.0, 200.0, 150.0],
			}
		)

	@patch("resources.graphs.st")
	def test_product_month_renders(
		self, mock_st: MagicMock, sales_df: pl.DataFrame
	) -> None:
		"""
		Verifies that product_month calls Streamlit functions.

		Returns:
			None.
		"""

		mock_st.columns.return_value = [MagicMock(), MagicMock()]
		product_month(sales_df, 2024)
		mock_st.subheader.assert_called_once()


class TestGenderStatus:
	"""
	Tests for the gender_status function.
	"""

	@pytest.fixture
	def sales_df(self) -> pl.DataFrame:
		"""
		Provides a DataFrame with sales data for testing.

		Returns:
			A DataFrame with sale dates, gender, and product status.
		"""

		return pl.DataFrame(
			{
				COL_FECHA_VENTA: [date(2024, 1, 15), date(2024, 1, 20)],
				COL_GENERO: ["F", "M"],
				COL_ESTADO_PRODUCTO: ["New", "Used"],
			}
		)

	@patch("resources.graphs.st")
	def test_gender_status_renders(
		self, mock_st: MagicMock, sales_df: pl.DataFrame
	) -> None:
		"""
		Verifies that gender_status calls Streamlit functions.

		Returns:
			None.
		"""

		gender_status(sales_df, 2024)
		mock_st.subheader.assert_called_once()
		mock_st.plotly_chart.assert_called_once()


class TestStatusCountry:
	"""
	Tests for the status_country function.
	"""

	@pytest.fixture
	def sales_df(self) -> pl.DataFrame:
		"""
		Provides a DataFrame with sales data for testing.

		Returns:
			A DataFrame with sale dates, country, and product status.
		"""

		return pl.DataFrame(
			{
				COL_FECHA_VENTA: [date(2024, 1, 15), date(2024, 1, 20)],
				COL_PAIS: ["Spain", "France"],
				COL_ESTADO_PRODUCTO: ["New", "Used"],
			}
		)

	@patch("resources.graphs.st")
	def test_status_country_renders(
		self, mock_st: MagicMock, sales_df: pl.DataFrame
	) -> None:
		"""
		Verifies that status_country calls Streamlit functions.

		Returns:
			None.
		"""

		status_country(sales_df, 2024)
		mock_st.subheader.assert_called_once()
		mock_st.plotly_chart.assert_called_once()


class TestGenderCountry:
	"""
	Tests for the gender_country function.
	"""

	@pytest.fixture
	def sales_df(self) -> pl.DataFrame:
		"""
		Provides a DataFrame with sales data for testing.

		Returns:
			A DataFrame with sale dates, country, and gender.
		"""

		return pl.DataFrame(
			{
				COL_FECHA_VENTA: [date(2024, 1, 15), date(2024, 1, 20)],
				COL_PAIS: ["Spain", "France"],
				COL_GENERO: ["F", "M"],
			}
		)

	@patch("resources.graphs.st")
	def test_gender_country_renders(
		self, mock_st: MagicMock, sales_df: pl.DataFrame
	) -> None:
		"""
		Verifies that gender_country calls Streamlit functions.

		Returns:
			None.
		"""

		mock_st.columns.return_value = [MagicMock(), MagicMock()]
		gender_country(sales_df, 2024)
		mock_st.subheader.assert_called_once()


class TestCompareProductsYears:
	"""
	Tests for the compare_products_years function.
	"""

	@pytest.fixture
	def sales_df(self) -> pl.DataFrame:
		"""
		Provides a DataFrame with sales data for testing.

		Returns:
			A DataFrame with sale dates and product types.
		"""

		return pl.DataFrame(
			{
				COL_FECHA_VENTA: [
					date(2024, 1, 15),
					date(2023, 1, 20),
					date(2024, 3, 10),
				],
				COL_TIPO_PRODUCTO: ["A", "B", "A"],
			}
		)

	@patch("resources.graphs.st")
	def test_compare_products_years_renders(
		self, mock_st: MagicMock, sales_df: pl.DataFrame
	) -> None:
		"""
		Verifies that compare_products_years calls Streamlit
		functions.

		Returns:
			None.
		"""

		compare_products_years(sales_df, [2023, 2024])
		mock_st.subheader.assert_called_once()
		mock_st.plotly_chart.assert_called_once()


class TestCompareIncomeMonthYears:
	"""
	Tests for the compare_income_month_years function.
	"""

	@pytest.fixture
	def sales_df(self) -> pl.DataFrame:
		"""
		Provides a DataFrame with sales data for testing.

		Returns:
			A DataFrame with sale dates and product prices.
		"""

		return pl.DataFrame(
			{
				COL_FECHA_VENTA: [
					date(2024, 1, 15),
					date(2023, 1, 20),
					date(2024, 3, 10),
				],
				COL_PRECIO_PRODUCTO: [100.0, 200.0, 150.0],
			}
		)

	@patch("resources.graphs.st")
	def test_compare_income_month_years_renders(
		self, mock_st: MagicMock, sales_df: pl.DataFrame
	) -> None:
		"""
		Verifies that compare_income_month_years calls
		Streamlit functions.

		Returns:
			None.
		"""

		compare_income_month_years(sales_df, [2023, 2024])
		mock_st.subheader.assert_called_once()
		mock_st.plotly_chart.assert_called_once()


class TestFlowMoney:
	"""
	Tests for the flow_money function.
	"""

	@pytest.fixture
	def sales_df(self) -> pl.DataFrame:
		"""
		Provides a DataFrame with sales data for testing.

		Returns:
			A DataFrame with sale dates, product prices,
				and discounts.
		"""

		return pl.DataFrame(
			{
				COL_FECHA_VENTA: [
					date(2024, 1, 15),
					date(2023, 1, 20),
					date(2024, 3, 10),
				],
				COL_PRECIO_PRODUCTO: [100.0, 200.0, 150.0],
				COL_DESCUENTO_APLICADO: [10.0, 20.0, 5.0],
			}
		)

	@patch("resources.graphs.st")
	def test_flow_money_renders(
		self, mock_st: MagicMock, sales_df: pl.DataFrame
	) -> None:
		"""
		Verifies that flow_money calls Streamlit functions.

		Returns:
			None.
		"""

		mock_st.columns.return_value = [
			MagicMock(),
			MagicMock(),
			MagicMock(),
		]
		flow_money(sales_df, [2023, 2024])
		mock_st.subheader.assert_called_once()
		mock_st.dataframe.assert_called_once()


class TestCategorySalesByPeriod:
	"""
	Tests for the category_sales_by_period function.
	"""

	@pytest.fixture
	def sales_df(self) -> pl.DataFrame:
		"""
		Provides a DataFrame with sales data for testing.

		Returns:
			A DataFrame with sale dates, product types,
				and prices.
		"""

		return pl.DataFrame(
			{
				COL_FECHA_VENTA: [
					date(2024, 1, 15),
					date(2023, 1, 20),
					date(2024, 1, 10),
				],
				COL_TIPO_PRODUCTO: ["A", "B", "A"],
				COL_PRECIO_PRODUCTO: [100.0, 200.0, 150.0],
			}
		)

	@patch("resources.graphs.st")
	def test_category_sales_by_period_renders(
		self, mock_st: MagicMock, sales_df: pl.DataFrame
	) -> None:
		"""
		Verifies that category_sales_by_period calls
		Streamlit functions.

		Returns:
			None.
		"""

		mock_st.columns.return_value = [MagicMock(), MagicMock()]
		mock_st.selectbox.return_value = "January"
		category_sales_by_period(sales_df, [2023, 2024])
		mock_st.subheader.assert_called_once()
		mock_st.selectbox.assert_called_once()
