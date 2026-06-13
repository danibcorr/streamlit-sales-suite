# 3pps
import polars as pl

# Own modules
from config import REQUIRED_COLUMNS
from config.constants import COL_FECHA_VENTA


class SalesDataFrameSchema:
	"""
	Schema for validating sales DataFrame columns.
	"""

	@classmethod
	def validate(cls, df: pl.DataFrame) -> tuple[bool, str]:
		"""
		Validates that the DataFrame contains all required columns
		and correct data types.

		Args:
			df: The DataFrame to validate.

		Returns:
			A tuple of (is_valid, error_message). If valid,
				error_message is empty.
		"""

		missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

		if missing_columns:
			return (
				False,
				f"Missing columns: {', '.join(sorted(missing_columns))}",
			)

		if df.schema[COL_FECHA_VENTA] != pl.Date and not isinstance(
			df.schema[COL_FECHA_VENTA], pl.Datetime
		):
			return False, f"'{COL_FECHA_VENTA}' must be a date type"

		return True, ""
