# Standard libraries
from datetime import date

# 3pps
import polars as pl

# Own modules
from config import REQUIRED_COLUMNS
from config.constants import COL_FECHA_VENTA
from dataframe_schema import SalesDataFrameSchema


class TestSalesDataFrameSchema:
	"""
	Tests for the SalesDataFrameSchema.validate method.
	"""

	def test_valid_dataframe(self) -> None:
		"""
		Verifies that a valid DataFrame passes validation.

		Returns:
			None.
		"""

		data = {col: ["x"] for col in REQUIRED_COLUMNS}
		data[COL_FECHA_VENTA] = [date(2024, 1, 1)]
		df = pl.DataFrame(data)
		is_valid, msg = SalesDataFrameSchema.validate(df)
		assert is_valid
		assert msg == ""

	def test_missing_columns(self) -> None:
		"""
		Verifies that missing columns are reported.

		Returns:
			None.
		"""

		df = pl.DataFrame({"fake": [1]})
		is_valid, msg = SalesDataFrameSchema.validate(df)
		assert not is_valid
		assert "Missing columns" in msg

	def test_invalid_date_type(self) -> None:
		"""
		Verifies that a non-date column fails validation.

		Returns:
			None.
		"""

		data = {col: ["x"] for col in REQUIRED_COLUMNS}
		df = pl.DataFrame(data)
		is_valid, msg = SalesDataFrameSchema.validate(df)
		assert not is_valid
		assert "date" in msg
