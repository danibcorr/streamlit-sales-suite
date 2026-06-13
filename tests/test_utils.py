# Standard libraries
from datetime import date

# 3pps
import polars as pl
import pytest

# Own modules
from config.constants import (
	COL_DESCUENTO_APLICADO,
	COL_FECHA_VENTA,
	COL_MONTH_NAME,
	COL_PRECIO_PRODUCTO,
	COL_TOTAL_PRODUCTS,
	COL_TOTAL_REVENUE,
)
from utils.utils import obtain_top, summarize_year


class TestObtainTop:
	"""
	Tests for the obtain_top function.
	"""

	@pytest.fixture
	def sample_df(self) -> pl.DataFrame:
		"""
		Provides a DataFrame with repeated category values
		for frequency-based testing.

		Returns:
			A DataFrame with a single ``category`` column.
		"""

		return pl.DataFrame({"category": ["A", "B", "A", "C", "B", "A"]})

	def test_top_1(self, sample_df: pl.DataFrame) -> None:
		"""
		Verifies that requesting the top 1 value returns
		only the most frequent category.

		Returns:
			None.
		"""

		result = obtain_top(sample_df, top=1, column="category")
		assert result == ["A"]

	def test_top_2(self, sample_df: pl.DataFrame) -> None:
		"""
		Verifies that requesting the top 2 values returns
		the two most frequent categories in order.

		Returns:
			None.
		"""

		result = obtain_top(sample_df, top=2, column="category")
		assert result == ["A", "B"]

	def test_top_exceeds_unique_values(self, sample_df: pl.DataFrame) -> None:
		"""
		Verifies that requesting more top values than
		unique entries returns all unique values without
		error.

		Returns:
			None.
		"""

		result = obtain_top(sample_df, top=10, column="category")
		assert len(result) == 3

	def test_empty_dataframe(self) -> None:
		"""
		Verifies that an empty DataFrame produces an
		empty result list.

		Returns:
			None.
		"""

		df = pl.DataFrame({"category": []}).cast({"category": pl.Utf8})
		result = obtain_top(df, top=5, column="category")
		assert result == []


class TestSummarizeYear:
	"""
	Tests for the summarize_year function.
	"""

	@pytest.fixture
	def sales_df(self) -> pl.DataFrame:
		"""
		Provides a DataFrame with sales records spanning
		2024 (January and March) and 2023 (June) for
		year-filtering tests.

		Returns:
			A DataFrame with sale date, product price,
				and discount columns.
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
				COL_DESCUENTO_APLICADO: [10.0, 20.0, 5.0, 15.0],
			}
		)

	def test_monthly_has_12_rows(self, sales_df: pl.DataFrame) -> None:
		"""
		Verifies that the monthly breakdown always
		contains exactly 12 rows, one per month.

		Returns:
			None.
		"""

		monthly, _ = summarize_year(sales_df, 2024)
		assert monthly.height == 12

	def test_annual_total_revenue(self, sales_df: pl.DataFrame) -> None:
		"""
		Verifies that the annual total revenue matches
		the sum of all product prices for the year.

		Returns:
			None.
		"""

		_, annual = summarize_year(sales_df, 2024)
		assert annual[COL_TOTAL_REVENUE] == 450.0

	def test_annual_total_products(self, sales_df: pl.DataFrame) -> None:
		"""
		Verifies that the annual total products count
		matches the number of sales for the year.

		Returns:
			None.
		"""

		_, annual = summarize_year(sales_df, 2024)
		assert annual[COL_TOTAL_PRODUCTS] == 3

	def test_year_with_no_data(self, sales_df: pl.DataFrame) -> None:
		"""
		Verifies that a year with no sales returns zero
		for both revenue and product count.

		Returns:
			None.
		"""

		_, annual = summarize_year(sales_df, 2025)
		assert annual[COL_TOTAL_REVENUE] == 0
		assert annual[COL_TOTAL_PRODUCTS] == 0

	def test_monthly_month_names(self, sales_df: pl.DataFrame) -> None:
		"""
		Verifies that month name labels are correctly
		assigned to their corresponding rows.

		Returns:
			None.
		"""

		monthly, _ = summarize_year(sales_df, 2024)
		assert monthly.row(0, named=True)[COL_MONTH_NAME] == "January"
		assert monthly.row(2, named=True)[COL_MONTH_NAME] == "March"
