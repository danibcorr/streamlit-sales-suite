# Standard libraries
import json

# 3pps
import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Own modules
from dataframe_schema import SalesDataFrameSchema
from utils import generate_synthetic_data


def credentials_case() -> UploadedFile | None:
	"""_summary_

	Returns:
		UploadedFile | None: _description_
	"""

	if "credentials_uploaded" not in st.session_state:
		st.session_state.credentials_uploaded = False

	if not st.session_state.credentials_uploaded:
		st.sidebar.subheader("Load credentials")

		credentials_file = st.sidebar.file_uploader(
			"Upload the JSON file with credentials", type=["json"]
		)
	else:
		credentials_file = None

	return credentials_file


def synthetic_case() -> bool | None:
	"""_summary_

	Returns:
		bool | None: _description_
	"""

	using_synthetic_data = False

	if "synthetic_data" not in st.session_state:
		st.session_state.synthetic_data = False

	if not st.session_state.synthetic_data and st.sidebar.checkbox(
		"Want to test the app with synthetic data?"
	):
		using_synthetic_data = True

	return using_synthetic_data


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

	credentials_file = credentials_case()
	using_synthetic_data = synthetic_case()

	if credentials_file is not None:
		try:
			parsed_credentials = json.load(credentials_file)

			st.session_state.credentials = parsed_credentials
			st.session_state.credentials_uploaded = True

			st.sidebar.success("Credentials successfully uploaded", icon="✅")

			if "path" in parsed_credentials:
				df = pd.read_excel(parsed_credentials["path"])
				is_valid, error_msg = SalesDataFrameSchema.validate(df)

				if is_valid:
					st.session_state.dataframe = df
					st.sidebar.success("File successfully loaded", icon="✅")
				else:
					st.sidebar.error(f"Invalid DataFrame: {error_msg}", icon="⚠️")
			else:
				st.sidebar.error("Invalid credentials: Missing path key.", icon="⚠️")

		except ValueError as error:
			st.sidebar.error(f"Error reading the JSON file: {error}", icon="⚠️")

	elif st.session_state.credentials_uploaded:
		st.sidebar.info("Credentials already uploaded and loaded.")

	if using_synthetic_data:
		st.session_state.dataframe = generate_synthetic_data()
		st.session_state.credentials = "Synthetic"
		st.session_state.credentials_uploaded = True


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

	st.session_state.setdefault("credentials", None)
	st.session_state.setdefault("dataframe", None)

	upload_credentials()

	current_page = st.navigation(navigation_pages)
	current_page.run()


streamlit_configuration()
