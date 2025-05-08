# 3pps
import streamlit as st

# Own modules
from src.utils import config_streamlit_page


def discount_calculator():
    """
    A Streamlit-based interactive discount calculator.
    """

    # Input fields with improved layout
    with st.container():
        original_price = st.number_input(
            "Price before discount (€)", min_value=0.0, value=0.0, format="%.2f"
        )
        discount = st.slider(
            "Discount (%)", min_value=0, max_value=100, value=0, step=1
        )

        saving = original_price * (discount / 100)
        price_after_discount = original_price - saving

        # Display results in columns for better layout
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Final Price", value=f"€{price_after_discount:.2f}", border=True
            )
        with col2:
            st.metric(label="You Save", value=f"€{saving:.2f}", border=True)


# First call to the config page function
config_streamlit_page(page_name="Discount Calculator")

# Call the discount calculator function
discount_calculator()
