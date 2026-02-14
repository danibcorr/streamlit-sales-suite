# Standard libraries
from dataclasses import dataclass

# 3pps
import pandas as pd

# Own modules
from config import REQUIRED_COLUMNS


@dataclass(frozen=True)
class SalesDataFrameSchema:
	"""
	Schema for validating sales DataFrame columns.
	"""

	@classmethod
	def validate(cls, df: pd.DataFrame) -> tuple[bool, str]:
		"""
		Validates that the DataFrame contains all required columns.

		Args:
			df: The DataFrame to validate.

		Returns:
			A tuple of (is_valid, error_message). If valid,
				error_message is empty.
		"""

		missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

		if missing_columns:
			return False, f"Missing columns: {', '.join(sorted(missing_columns))}"

		return True, ""
