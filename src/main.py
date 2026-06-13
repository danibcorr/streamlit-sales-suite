# Standard libraries
import json
import os

# 3pps
import polars as pl
import streamlit as st

# Own modules
from config.constants import (
	COL_FECHA_VENTA,
	SESSION_CREDENTIALS,
	SESSION_CREDENTIALS_UPLOADED,
	SESSION_DATAFRAME,
)
from dataframe_schema import SalesDataFrameSchema
from utils import generate_synthetic_data

ALLOWED_DATA_DIR: str = os.environ.get("SALES_DATA_DIR", os.getcwd())


def load_synthetic_data() -> None:
	"""
	Generates and loads synthetic sales data into session
	state, marking credentials as uploaded.

	Returns:
		None.
	"""

	st.session_state[SESSION_DATAFRAME] = generate_synthetic_data()
	st.session_state[SESSION_CREDENTIALS] = "Synthetic"
	st.session_state[SESSION_CREDENTIALS_UPLOADED] = True
	st.sidebar.success("Synthetic data loaded successfully", icon="✅")


def load_excel_file(path: str) -> None:
	"""
	Loads and validates an Excel file from the specified path.
	Updates session state with the DataFrame if valid, otherwise
	displays an error message.

	Args:
		path: Path to the Excel file.

	Returns:
		None.
	"""

	df = pl.read_excel(path).with_columns(pl.col(COL_FECHA_VENTA).cast(pl.Date))
	is_valid, error_msg = SalesDataFrameSchema.validate(df)

	if is_valid:
		st.session_state[SESSION_DATAFRAME] = df
		st.sidebar.success("File successfully loaded", icon="✅")
	else:
		st.sidebar.error(f"Invalid DataFrame: {error_msg}", icon="⚠️")


def process_credentials_file(credentials_file) -> None:
	"""
	Parses and processes a JSON credentials file. Updates
	session state with credentials and loads the Excel file
	specified in the credentials. Displays appropriate error
	messages for invalid input.

	Args:
		credentials_file: Uploaded JSON file object.

	Returns:
		None.
	"""

	try:
		parsed_credentials = json.load(credentials_file)

		st.session_state[SESSION_CREDENTIALS] = parsed_credentials
		st.session_state[SESSION_CREDENTIALS_UPLOADED] = True

		st.sidebar.success("Credentials successfully uploaded", icon="✅")

		if "path" in parsed_credentials:
			load_excel_file(parsed_credentials["path"])
		else:
			st.sidebar.error("Invalid credentials: Missing path key.", icon="⚠️")

	except json.JSONDecodeError as error:
		st.sidebar.error(f"Invalid JSON format: {error}", icon="⚠️")
	except KeyError as error:
		st.sidebar.error(f"Missing required key: {error}", icon="⚠️")
	except Exception as error:
		st.sidebar.error(f"Unexpected error: {error}", icon="⚠️")


def upload_credentials() -> None:
	"""
	Renders a sidebar file uploader for a JSON credentials
	file. On upload, parses the JSON, stores the credentials
	in Streamlit session state, and loads the Excel file
	referenced by the ``path`` key into a DataFrame stored
	in session state. Skips the uploader if credentials have
	already been loaded.

	Returns:
		None.
	"""

	if SESSION_CREDENTIALS_UPLOADED not in st.session_state:
		st.session_state[SESSION_CREDENTIALS_UPLOADED] = False

	if st.session_state[SESSION_CREDENTIALS_UPLOADED]:
		st.sidebar.info("Credentials already uploaded and loaded.")
		return

	st.sidebar.subheader("Load credentials")

	if st.sidebar.checkbox("Want to test the app with synthetic data?"):
		load_synthetic_data()
		return

	credentials_file = st.sidebar.file_uploader(
		"Upload the JSON file with credentials", type=["json"]
	)

	if credentials_file is not None:
		process_credentials_file(credentials_file)


def streamlit_configuration() -> None:
	"""
	Initializes the Streamlit application by setting page
	metadata, rendering the sidebar with project information,
	registering available pages, and handling credential
	upload and navigation.

	Returns:
		None.
	"""

	st.set_page_config(
		page_title="Streamlit Sales Suite",
		page_icon=":material/insert_chart:",
		layout="wide",
	)

	st.logo(
		image="src/images/logo.png",
		link="https://github.com/danibcorr/streamlit-sales-suite",
		size="large",
	)

	st.sidebar.subheader("About")
	st.sidebar.image(
		image="src/images/nikola-duza-fi6kmznklGQ-unsplash.jpg",
		width="stretch",
	)
	st.sidebar.markdown(
		"""
        This project simplifies analyzing product sales data for second-hand markets.
        While the data isn't publicly available, the code is licensed under the MIT
        License 🌻.
        """
	)
	st.sidebar.divider()

	navigation_pages = {
		"Resources": [
			st.Page(
				"./resources/graphs.py",
				title="Graphs",
				icon=":material/monitoring:",
			),
			st.Page(
				"./resources/discount_calculator.py",
				title="Discount Calculator",
				icon=":material/calculate:",
			),
		]
	}

	st.session_state.setdefault(SESSION_CREDENTIALS, None)
	st.session_state.setdefault(SESSION_DATAFRAME, None)

	upload_credentials()

	current_page = st.navigation(navigation_pages)
	current_page.run()


streamlit_configuration()
