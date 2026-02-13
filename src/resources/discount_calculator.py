# 3pps
import streamlit as st

# Own modules
from utils import config_streamlit_page


def discount_calculator() -> None:
	"""
	Renders an interactive discount calculator that accepts
	an original price and discount percentage, then displays
	the final price and total savings as Streamlit metrics.

	Returns:
		None.
	"""

	with st.container():
		original_price: float = st.number_input(
			"Price before discount (€)", min_value=0.0, value=0.0, format="%.2f"
		)
		discount: int = st.slider(
			"Discount (%)", min_value=0, max_value=100, value=0, step=1
		)

		saving: float = original_price * (discount / 100)
		price_after_discount: float = original_price - saving

		col1, col2 = st.columns(2)
		with col1:
			st.metric(
				label="Final Price", value=f"€{price_after_discount:.2f}", border=True
			)
		with col2:
			st.metric(label="You Save", value=f"€{saving:.2f}", border=True)


config_streamlit_page(page_name="Discount Calculator")
discount_calculator()
