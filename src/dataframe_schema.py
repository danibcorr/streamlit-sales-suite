# Standard libraries
from dataclasses import dataclass
from typing import Final

# 3pps
import pandas as pd


@dataclass(frozen=True)
class SalesDataFrameSchema:
	"""
	Schema for validating sales DataFrame columns.
	"""

	REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
		"Plataforma de venta",
		"Fecha de venta",
		"Pais",
		"Genero",
		"Precio producto",
		"Tipo producto",
		"Estado del producto",
		"Descuentos (%)",
	)

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

		missing_columns = set(cls.REQUIRED_COLUMNS) - set(df.columns)

		if missing_columns:
			return False, f"Missing columns: {', '.join(sorted(missing_columns))}"

		return True, ""
