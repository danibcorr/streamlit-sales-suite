# Standard libraries
from dataclasses import dataclass

# 3pps
import pandas as pd

# Own modules
from config import REQUIRED_COLUMNS
from config.constants import COL_FECHA_VENTA


@dataclass(frozen=True)
class SalesDataFrameSchema:
	"""
	Schema for validating sales DataFrame columns.
	"""

	@classmethod
	def validate(cls, df: pd.DataFrame) -> tuple[bool, str]:
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
			return False, f"Missing columns: {', '.join(sorted(missing_columns))}"

		if not pd.api.types.is_datetime64_any_dtype(df[COL_FECHA_VENTA]):
			return False, f"'{COL_FECHA_VENTA}' must be datetime type"

		return True, ""
