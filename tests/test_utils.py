# Standard libraries

# 3pps
import pandas as pd
import pytest

# Own modules
from utils.utils import obtain_top, summarize_year


class TestObtainTop:
	"""
	Tests for the obtain_top function.
	"""

	@pytest.fixture
	def sample_df(self) -> pd.DataFrame:
		"""
		Provides a DataFrame with repeated category values
		for frequency-based testing.

		Returns:
			A DataFrame with a single ``category`` column.
		"""

		return pd.DataFrame({"category": ["A", "B", "A", "C", "B", "A"]})

	def test_top_1(self, sample_df: pd.DataFrame) -> None:
		"""
		Verifies that requesting the top 1 value returns
		only the most frequent category.

		Returns:
			None.
		"""

		result = obtain_top(sample_df, top=1, column="category")
		assert result == ["A"]

	def test_top_2(self, sample_df: pd.DataFrame) -> None:
		"""
		Verifies that requesting the top 2 values returns
		the two most frequent categories in order.

		Returns:
			None.
		"""

		result = obtain_top(sample_df, top=2, column="category")
		assert result == ["A", "B"]

	def test_top_exceeds_unique_values(self, sample_df: pd.DataFrame) -> None:
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

		df = pd.DataFrame({"category": []})
		result = obtain_top(df, top=5, column="category")
		assert result == []


class TestSummarizeYear:
	"""
	Tests for the summarize_year function.
	"""

	@pytest.fixture
	def sales_df(self) -> pd.DataFrame:
		"""
		Provides a DataFrame with sales records spanning
		2024 (January and March) and 2023 (June) for
		year-filtering tests.

		Returns:
			A DataFrame with sale date, product price,
				and discount columns.
		"""

		return pd.DataFrame(
			{
				"Fecha de venta": pd.to_datetime(
					["2024-01-15", "2024-01-20", "2024-03-10", "2023-06-01"]
				),
				"Precio producto": [100.0, 200.0, 150.0, 50.0],
				"Descuentos (%)": [10.0, 20.0, 5.0, 15.0],
			}
		)

	def test_monthly_has_12_rows(self, sales_df: pd.DataFrame) -> None:
		"""
		Verifies that the monthly breakdown always
		contains exactly 12 rows, one per month.

		Returns:
			None.
		"""

		monthly, _ = summarize_year(sales_df, 2024)
		assert len(monthly) == 12

	def test_annual_total_revenue(self, sales_df: pd.DataFrame) -> None:
		"""
		Verifies that the annual total revenue matches
		the sum of all product prices for the year.

		Returns:
			None.
		"""

		_, annual = summarize_year(sales_df, 2024)
		assert annual["Total_Revenue"] == 450.0

	def test_annual_total_products(self, sales_df: pd.DataFrame) -> None:
		"""
		Verifies that the annual total products count
		matches the number of sales for the year.

		Returns:
			None.
		"""

		_, annual = summarize_year(sales_df, 2024)
		assert annual["Total_Products"] == 3

	def test_year_with_no_data(self, sales_df: pd.DataFrame) -> None:
		"""
		Verifies that a year with no sales returns zero
		for both revenue and product count.

		Returns:
			None.
		"""

		monthly, annual = summarize_year(sales_df, 2025)
		assert annual["Total_Revenue"] == 0
		assert annual["Total_Products"] == 0

	def test_monthly_month_names(self, sales_df: pd.DataFrame) -> None:
		"""
		Verifies that month name labels are correctly
		assigned to their corresponding rows.

		Returns:
			None.
		"""

		monthly, _ = summarize_year(sales_df, 2024)
		assert monthly.loc[0, "Month_Name"] == "January"
		assert monthly.loc[2, "Month_Name"] == "March"
