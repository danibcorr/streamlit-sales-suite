# Standard libraries
import time

# 3pps
import pandas as pd
import streamlit as st


def config_streamlit_page(page_name: str) -> None:
	"""
	Configures the Streamlit page settings based on the given page name.

	Args:
		page_name (str): The name of the page that is being configured.
	"""

	# Configure the title of the Streamlit page
	st.title(page_name)


def check_credentials() -> bool:
	"""
	Checks whether credentials are available in the session state.

	This function verifies if the credentials are stored in the Streamlit session state.
	If available, it shows a success message for 3 seconds, and returns True.
	If not available, it shows a warning message and returns False.

	Returns:
		bool: True if credentials are available, otherwise False.
	"""

	@st.cache_data
	def credentials_available():
		# Create an empty container that can be updated
		message_container = st.empty()

		# Show a success message
		message_container.success("Credentials available.", icon="🔐")

		# Wait for 3 seconds
		time.sleep(3)

		# Empty the message
		message_container.empty()

	if "credentials" in st.session_state and st.session_state.credentials:
		credentials_available()
		return True

	st.warning("Credentials not available.", icon="⚠️")
	return False


def obtain_top(df: pd.DataFrame, top: int, column: str) -> list:
	"""Obtain the top 'n' rows from a DataFrame based on a specific column.

	Args:
		df: The DataFrame containing the data.
		top: The number of top rows to return based on the column value.
		column: The column name to sort the data by to determine the top rows.

	Returns:
		list: A list of the top 'n' values from the specified column in the DataFrame.
	"""

	return list(df[column].value_counts()[:top].index)
