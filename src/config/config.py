# Standard libraries
from typing import Final

MONTHS_NAMES: Final[tuple[str, ...]] = (
	"January",
	"February",
	"March",
	"April",
	"May",
	"June",
	"July",
	"August",
	"September",
	"October",
	"November",
	"December",
)

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
