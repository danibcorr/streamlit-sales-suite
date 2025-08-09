# Standard libraries
import json

# 3pps
import pandas as pd
import streamlit as st


def upload_credentials() -> None:
	"""
	Handles the uploading of a JSON file containing credentials and stores them
	in the session state. It also loads a corresponding Excel file based on the
	credentials and stores the resulting DataFrame in the session state.
	"""

	# Check if credentials have already been uploaded and set a flag in session_state
	if "credentials_uploaded" not in st.session_state:
		st.session_state.credentials_uploaded = False

	# Input to upload a JSON file
	if not st.session_state.credentials_uploaded:
		# Title displayed to upload credentials
		st.sidebar.subheader("Load credentials")

		uploaded_file = st.sidebar.file_uploader(
			"Upload the JSON file with credentials", type=["json"]
		)
	else:
		uploaded_file = None  # Disable the file uploader if credentials are uploaded

	# Button to save credentials and load data if JSON is uploaded
	if uploaded_file is not None:
		try:
			# Read JSON file
			credentials = json.load(uploaded_file)

			# Save in session_state the value of credentials
			st.session_state.credentials = credentials
			st.session_state.credentials_uploaded = True

			# Show the success message in the container
			st.sidebar.success("Credentials successfully uploaded", icon="✅")

			# Attempt to load the file if path is valid
			if "path" in credentials:
				st.session_state.dataframe = pd.read_excel(credentials["path"])
				st.sidebar.success("File successfully loaded", icon="✅")
			else:
				st.sidebar.error("Invalid credentials: Missing path key.", icon="⚠️")

		except ValueError as e:
			st.sidebar.error(f"Error reading the JSON file: {e}", icon="⚠️")
	else:
		# Provide feedback if the user has already uploaded the credentials
		if st.session_state.credentials_uploaded:
			st.sidebar.info("Credentials already uploaded and loaded.")


def streamlit_configuration() -> None:
	"""
	Configures the Streamlit app with page settings, project resources, and credentials
	handling.
	"""

	# Page configuration for the project
	st.set_page_config(
		page_title="Streamlit Sales Suite",
		page_icon=":material/insert_chart:",
		layout="wide",
	)

	# Logo for the project
	st.logo(
		image="./docs/imgs/logo.png",
		link="https://github.com/danibcorr/streamlit-sales-suite",
		size="large",
	)

	# Add information related to the project
	st.sidebar.subheader("About")
	st.sidebar.image(
		image="./docs/imgs/nikola-duza-fi6kmznklGQ-unsplash.jpg",
		use_container_width=True,
	)
	st.sidebar.markdown(
		"""
        This project simplifies analyzing product sales data for second-hand markets.
        While the data isn’t publicly available, the code is licensed under the MIT
        License 🌻.
        """
	)
	st.sidebar.divider()

	# Pages used for the project
	pages = {
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
			st.Page(
				"./resources/interact.py",
				title="Interact",
				icon=":material/interactive_space:",
			),
		]
	}

	# Initialize credentials and dataframe in session_state with default values
	st.session_state.setdefault("credentials", None)
	st.session_state.setdefault("dataframe", None)

	# Button for uploading credentials for access to the data
	upload_credentials()

	# Execution of the page configuration
	pg = st.navigation(pages)
	pg.run()


# Load the streamlit app configuration
streamlit_configuration()
