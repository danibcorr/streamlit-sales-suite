# Standard libraries
from typing import Final

# Session state keys
SESSION_CREDENTIALS: Final[str] = "credentials"
SESSION_DATAFRAME: Final[str] = "dataframe"
SESSION_CREDENTIALS_UPLOADED: Final[str] = "credentials_uploaded"

# Column names - Original data
COL_FECHA_VENTA: Final[str] = "Fecha de venta"
COL_PRECIO_PRODUCTO: Final[str] = "Precio producto"
COL_DESCUENTOS: Final[str] = "Descuentos (%)"
COL_TIPO_PRODUCTO: Final[str] = "Tipo producto"
COL_GENERO: Final[str] = "Genero"
COL_PAIS: Final[str] = "Pais"
COL_ESTADO_PRODUCTO: Final[str] = "Estado del producto"
COL_PLATAFORMA: Final[str] = "Plataforma de venta"

# Column names - Computed/Intermediate
COL_MONTH_ES: Final[str] = "Mes"
COL_MONTH: Final[str] = "Month"
COL_MONTH_NAME: Final[str] = "Month_Name"
COL_MONTH_NAME_ES: Final[str] = "Mes_nombre"
COL_YEAR_MONTH: Final[str] = "Año_Mes"
COL_YEAR: Final[str] = "Year"
COL_YEAR_ES: Final[str] = "Año"
COL_REVENUE: Final[str] = "Revenue"
COL_PRODUCTS_SOLD: Final[str] = "Products_Sold"
COL_AVG_DISCOUNT: Final[str] = "Avg_Discount"
COL_AVG_PRICE: Final[str] = "Avg_Price"
COL_TOTAL_REVENUE: Final[str] = "Total_Revenue"
COL_TOTAL_PRODUCTS: Final[str] = "Total_Products"
COL_QUANTITY_ES: Final[str] = "Cantidad"
COL_TOTAL_PRICE: Final[str] = "Total_Price"
COL_QUANTITY: Final[str] = "Quantity"
COL_MONEY: Final[str] = "Dinero"
COL_COUNT: Final[str] = "count"
