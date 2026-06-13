# Standard libraries
from typing import Final

# Session state keys
SESSION_CREDENTIALS: Final[str] = "credentials"
SESSION_DATAFRAME: Final[str] = "dataframe"
SESSION_CREDENTIALS_UPLOADED: Final[str] = "credentials_uploaded"
SESSION_SYNTHETIC_DATA: Final[str] = "synthetic_data"

# Column names - Original data (Spanish, matching input file)
COL_PLATAFORMA: Final[str] = "Plataforma de venta"
COL_FECHA_VENTA: Final[str] = "Fecha de venta"
COL_PAIS: Final[str] = "Pais"
COL_GENERO: Final[str] = "Genero"
COL_COSTE_ADQUISICION: Final[str] = "Coste adquisición"
COL_DESCUENTO_APLICADO: Final[str] = "Descuento aplicado"
COL_DESCUENTO_LOTE: Final[str] = "Descuentos lote (%)"
COL_PRECIO_PRODUCTO: Final[str] = "Precio producto"
COL_TIPO_PRODUCTO: Final[str] = "Tipo producto"
COL_ESTADO_PRODUCTO: Final[str] = "Estado del producto"
COL_FAVORITOS: Final[str] = "Favoritos"
COL_VISITAS: Final[str] = "Visitas"

# Column names - Computed/Intermediate (English)
COL_MONTH: Final[str] = "Month"
COL_MONTH_NUM: Final[str] = "Month_Num"
COL_MONTH_NAME: Final[str] = "Month_Name"
COL_YEAR: Final[str] = "Year"
COL_YEAR_STR: Final[str] = "Year_Str"
COL_YEAR_MONTH: Final[str] = "Year_Month"
COL_REVENUE: Final[str] = "Revenue"
COL_PRODUCTS_SOLD: Final[str] = "Products_Sold"
COL_AVG_DISCOUNT: Final[str] = "Avg_Discount"
COL_AVG_PRICE: Final[str] = "Avg_Price"
COL_TOTAL_REVENUE: Final[str] = "Total_Revenue"
COL_TOTAL_PRODUCTS: Final[str] = "Total_Products"
COL_TOTAL_PRICE: Final[str] = "Total_Price"
COL_QUANTITY: Final[str] = "Quantity"
COL_INCOME: Final[str] = "Income"
COL_COUNT: Final[str] = "count"
