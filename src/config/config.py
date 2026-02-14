# Standard libraries
from typing import Final

# Own modules
from config.constants import (
    COL_DESCUENTOS,
    COL_ESTADO_PRODUCTO,
    COL_FECHA_VENTA,
    COL_GENERO,
    COL_PAIS,
    COL_PLATAFORMA,
    COL_PRECIO_PRODUCTO,
    COL_TIPO_PRODUCTO,
)

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
	COL_PLATAFORMA,
	COL_FECHA_VENTA,
	COL_PAIS,
	COL_GENERO,
	COL_PRECIO_PRODUCTO,
	COL_TIPO_PRODUCTO,
	COL_ESTADO_PRODUCTO,
	COL_DESCUENTOS,
)
