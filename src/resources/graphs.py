# 3pps
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Own modules
from config import MONTHS_NAMES
from utils import check_credentials, config_streamlit_page, obtain_top, summarize_year


def money_month(df: pd.DataFrame, year: int) -> None:
	"""
	Calculates and visualizes the total money earned per month for a given year.

	Args:
		df: The DataFrame containing sales data.
		year: The selected year.
	"""

	# Title of the plot
	st.subheader(f"Money Earned per Month in {year}")

	# First we filter by year
	df_filtered = df[df["Fecha de venta"].dt.year == year].copy()

	# Then we create a column by month
	df_filtered["Mes"] = df_filtered["Fecha de venta"].dt.month

	# We group the money obtained per month
	df_money_month = df_filtered.groupby("Mes")["Precio producto"].sum()

	# We fill in those months without data with values at 0
	all_months = range(1, 13)
	df_money_month = df_money_month.reindex(all_months, fill_value=0)

	# Convert to DataFrame and include month names
	df_money_month = pd.DataFrame(
		{"Mes": MONTHS_NAMES, "Dinero": df_money_month.values}
	)

	# Ensure correct order with pd.Categorical
	df_money_month["Mes"] = pd.Categorical(
		df_money_month["Mes"], categories=MONTHS_NAMES, ordered=True
	)

	# Create a bar chart
	fig = px.bar(
		df_money_month,
		x="Mes",
		y="Dinero",
		labels={"Dinero": "Income (€)", "Mes": "Month"},
		color="Mes",
		color_discrete_sequence=px.colors.qualitative.Pastel,
	)

	# Update layout
	fig.update_layout(
		xaxis_title="Month",
		yaxis_title="Income (€)",
		xaxis_tickangle=-45,
		showlegend=False,
		coloraxis_showscale=False,
	)

	# Add value labels on bars
	fig.update_traces(texttemplate="%{y:.0f}€", textposition="outside")

	# Display the plot in Streamlit
	st.plotly_chart(fig, use_container_width=True)


def product_month(df: pd.DataFrame, year: int) -> None:
	"""
	Plot information per month, like products sold, top 3, and more.

	Args:
		df: The DataFrame containing sales data.
		year: The selected year.
	"""

	# Title of the plot
	st.subheader(f"Products Sold per Month in {year}")

	# Create columns for the plots
	col1, col2 = st.columns(2, gap="large")

	# Filter data by year
	df_filtered_year = df[df["Fecha de venta"].dt.year == year].copy()
	df_filtered = df_filtered_year.copy()
	df_filtered["Mes"] = df_filtered_year["Fecha de venta"].dt.month
	df_filtered = (
		df_filtered.groupby(["Mes", "Tipo producto"]).size().reset_index(name="count")
	)

	# Convert month numbers to month names
	df_filtered["Mes"] = df_filtered["Mes"].apply(lambda x: MONTHS_NAMES[x - 1])

	# Ensure correct order with pd.Categorical
	df_filtered["Mes"] = pd.Categorical(
		df_filtered["Mes"], categories=MONTHS_NAMES, ordered=True
	)

	# Pivot for plotting
	df_pivotado = df_filtered.pivot(
		index="Mes", columns="Tipo producto", values="count"
	).fillna(0)

	# Remove months with no products
	df_pivotado = df_pivotado.loc[(df_pivotado > 0).any(axis=1)]

	with col1:
		# Create horizontal bar chart with Plotly
		fig = go.Figure()

		# Add a bar for each product type
		for product_type in df_pivotado.columns:
			fig.add_trace(
				go.Bar(
					y=df_pivotado.index,
					x=df_pivotado[product_type],
					name=product_type,
					orientation="h",
				)
			)

		fig.update_layout(
			xaxis_title="Number of Products",
			yaxis_title="Month",
			barmode="stack",
			height=400,
			legend=dict(
				orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
			),
		)

		# Display the plot in Streamlit
		st.plotly_chart(fig, use_container_width=True)

	with col2:
		top_3_products = obtain_top(df=df_filtered_year, top=3, column="Tipo producto")
		st.metric(
			label="Top 3 Products", value=str(", ".join(top_3_products)), border=True
		)

		a, b = st.columns(2)
		a.metric(
			label="Total Products Sold",
			value=int(df_filtered_year["Tipo producto"].value_counts().sum()),
			border=True,
		)
		b.metric(
			label="Total Cash Obtained",
			value=str(float(df_filtered_year["Precio producto"].sum().round(4))) + " €",
			border=True,
		)


def interest_category(df: pd.DataFrame, year: int) -> None:
	"""
	Analyzes and visualizes the number of likes per product category for a given year.

	Args:
		df: The DataFrame containing sales data.
		year: The selected year.
	"""

	st.subheader(f"Likes by Category in {year}")

	# Filter data for the selected year
	df_filtered = df[df["Fecha de venta"].dt.year == year].copy()

	# Group by 'Tipo producto' and count 'Numero de mg'
	df_filtered = df_filtered.groupby("Tipo producto")["Numero de mg"].count()

	# Convert the result to a DataFrame
	df_filtered = pd.DataFrame(
		{"Tipo producto": df_filtered.index, "Numero de mg": df_filtered.values}
	)

	# Sort by likes in descending order for better visualization
	df_filtered = df_filtered.sort_values("Numero de mg", ascending=False)

	# Create bar chart with Plotly
	fig = px.bar(
		df_filtered,
		x="Tipo producto",
		y="Numero de mg",
		labels={"Tipo producto": "Product Type", "Numero de mg": "Number of Likes"},
		color="Tipo producto",
		color_continuous_scale=px.colors.qualitative.Pastel,
	)

	# Update layout for better appearance
	fig.update_layout(
		xaxis_title="Product Type",
		yaxis_title="Number of Likes",
		xaxis_tickangle=-45,
		showlegend=False,
		height=500,
	)

	# Add value labels on top of bars
	fig.update_traces(texttemplate="%{y}", textposition="outside")

	# Display the plot in Streamlit
	st.plotly_chart(fig, use_container_width=True)


def gender_status(df: pd.DataFrame, year: int) -> None:
	"""
	Calculate and visualize the gender distribution by product status for a given year.

	Args:
		df: The DataFrame containing sales data.
		year: The selected year.
	"""

	st.subheader(f"Gender Status Heatmap in {year}")

	# Filter data by the selected year
	data_year = df[df["Fecha de venta"].dt.year == year].copy()

	# Group by gender and product status
	df_2dhist = data_year.pivot_table(
		index="Estado del producto",
		columns="Genero",
		values="Fecha de venta",
		aggfunc="count",
		fill_value=0,
	)

	# Create the heatmap using Plotly
	fig = px.imshow(
		df_2dhist,
		text_auto=True,
		color_continuous_scale="mint",
		labels={"color": "Count"},
		aspect="auto",
	)

	# Adjust the layout to remove the grid, axes background transparency, and colorbar
	fig.update_layout(
		xaxis_title="Gender", yaxis_title="Product Status", coloraxis_showscale=False
	)

	# Display the plot in Streamlit
	st.plotly_chart(fig, use_container_width=True)


def status_country(df: pd.DataFrame, year: int) -> None:
	"""
	Calculate and visualize the product status distribution by country for a given year.

	Args:
		df: The DataFrame containing sales data.
		year: The selected year.
	"""

	st.subheader(f"Product Status by Country Heatmap in {year}")

	# Filter data for the specified year
	data_year = df[df["Fecha de venta"].dt.year == year].copy()

	# Group by product status and count the occurrences by country
	df_2dhist = data_year.pivot_table(
		index="Pais",
		columns="Estado del producto",
		values="Fecha de venta",
		aggfunc="count",
		fill_value=0,
	)

	# Create the heatmap using Plotly
	fig = px.imshow(
		df_2dhist,
		text_auto=True,
		color_continuous_scale="mint",
		labels={"color": "Count"},
		aspect="auto",
	)

	# Adjust the layout to remove the grid, axes background transparency, and colorbar
	fig.update_layout(
		xaxis_title="Product Status",
		yaxis_title="Country",
		# Hide the colorbar
		coloraxis_showscale=False,
	)

	# Display the plot in Streamlit
	st.plotly_chart(fig, use_container_width=True)


def gender_country(df: pd.DataFrame, year: int) -> None:
	"""
	Calculate and visualize the gender distribution by country for a given year.

	Args:
		df: The DataFrame containing sales data.
		year: The selected year.
	"""

	st.subheader(f"Gender Distribution by Country in {year}")

	# Filter data for the specified year
	data_year = df[df["Fecha de venta"].dt.year == year].copy()

	# Group by product status and count the occurrences by country
	df_2dhist = data_year.pivot_table(
		index="Pais",
		columns="Genero",
		values="Fecha de venta",
		aggfunc="count",
		fill_value=0,
	)

	# Split in genres
	df_f = df_2dhist["F"].sort_values(ascending=False).to_frame()
	df_m = df_2dhist["M"].sort_values(ascending=False).to_frame()

	col1, col2 = st.columns(2)
	with col1:
		# Create the heatmap using Plotly
		fig = px.imshow(
			df_f,
			text_auto=True,
			color_continuous_scale="mint",
			labels={"color": "Count"},
			aspect="auto",
		)

		# Adjust the layout to remove the grid, axes background transparency
		# and colorbar
		fig.update_layout(
			xaxis_title="Gender",
			yaxis_title="Country",
			coloraxis_showscale=False,
		)
		st.plotly_chart(fig, use_container_width=True)
	with col2:
		# Create the heatmap using Plotly
		fig = px.imshow(
			df_m,
			text_auto=True,
			color_continuous_scale="mint",
			labels={"color": "Count"},
			aspect="auto",
		)

		# Adjust the layout to remove the grid, axes background transparency,
		# and colorbar
		fig.update_layout(
			xaxis_title="Gender",
			yaxis_title="Country",
			coloraxis_showscale=False,
		)
		st.plotly_chart(fig, use_container_width=True)


def compare_products_years(df: pd.DataFrame, years: list[int]) -> None:
	"""
	Compare product sold by type and year.

	Args:
		df: The DataFrame containing sales data.
		years: The selected years.
	"""

	st.subheader(
		"Comparison of Products Sold by Type and Year in "
		f"{str(', '.join([str(year) for year in years]))}"
	)

	# First we filter the data for the years selected
	data_filtered = df[df["Fecha de venta"].dt.year.isin(years)]

	df_pivot = (
		data_filtered.groupby(
			["Tipo producto", data_filtered["Fecha de venta"].dt.year]
		)
		.size()
		.reset_index(name="Cantidad")
	)
	df_pivot.columns = ["Tipo producto", "Año", "Cantidad"]
	df_pivot["Año"] = df_pivot["Año"].astype(str)

	fig = px.bar(
		df_pivot,
		x="Tipo producto",
		y="Cantidad",
		color="Año",
		title="",
		barmode="group",
		text_auto=True,
		color_discrete_sequence=px.colors.qualitative.Pastel,
	)
	fig.update_layout(
		xaxis_title="Product Type",
		yaxis_title="Quantity",
		legend_title="Year",
		barmode="group",
		xaxis_tickangle=-45,
	)

	fig.update_traces(
		textfont_size=12, textangle=0, textposition="outside", cliponaxis=False
	)

	st.plotly_chart(fig, use_container_width=True)


def compare_income_month_years(df: pd.DataFrame, years: list[int]) -> None:
	"""
	Compare income per month for multiple years.

	Args:
		df: The DataFrame containing sales data.
		years: The selected years.
	"""

	st.subheader(
		"Comparison of Income by Month and Year in "
		f"{str(', '.join([str(year) for year in years]))}"
	)

	# First we filter the data for the years selected
	data_filtered = df[df["Fecha de venta"].dt.year.isin(years)]

	data_filtered["Fecha de venta"] = pd.to_datetime(data_filtered["Fecha de venta"])

	data_filtered["Año"] = data_filtered["Fecha de venta"].dt.year.astype(str)
	data_filtered["Mes"] = data_filtered["Fecha de venta"].dt.month
	data_filtered["Mes_nombre"] = data_filtered["Fecha de venta"].dt.strftime("%B")
	data_filtered["Año_Mes"] = data_filtered["Fecha de venta"].dt.strftime("%Y-%m")

	df_ganancias = (
		data_filtered.groupby(["Año", "Mes", "Mes_nombre", "Año_Mes"])[
			"Precio producto"
		]
		.sum()
		.reset_index()
	)

	fig = px.bar(
		df_ganancias,
		x="Mes_nombre",
		y="Precio producto",
		color="Año",
		barmode="group",
		text="Precio producto",
		color_discrete_sequence=px.colors.qualitative.Pastel,
		category_orders={
			"Mes_nombre": [
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
			]
		},
	)

	fig.update_layout(
		xaxis_title="Month",
		yaxis_title="Income (€)",
		legend_title="Year",
		xaxis_tickangle=-45,
	)

	fig.update_traces(
		texttemplate="%{text:.0f}€",
		textposition="outside",
		textfont_size=10,
		cliponaxis=False,
	)

	st.plotly_chart(fig, use_container_width=True)


def flow_money(df: pd.DataFrame, years: list[int]) -> None:
	"""
	Compara flujo de ingresos y métricas para varios años.

	Args:
		df: The DataFrame containing sales data.
		years: The selected years.
	"""

	st.subheader(
		f"Financial Flow Analysis: {str(', '.join([str(year) for year in years]))}"
	)

	monthly_all, annual_all = [], []

	for y in years:
		m, a = summarize_year(df, y)
		monthly_all.append(m)
		annual_all.append(a)

	annual_df = pd.DataFrame(annual_all)

	col1, col2, col3 = st.columns(3)

	col1.metric("Total Revenue", f"{annual_df['Total_Revenue'].sum():,.2f} €")
	col2.metric("Total Products", f"{annual_df['Total_Products'].sum():,}")
	col3.metric("Total Visits", f"{annual_df['Total_Visits'].sum():,}")

	st.dataframe(annual_df.round(2), use_container_width=True)


def display_all_graphs(credentials_status: bool) -> None:
	"""
	Displays various graphs based on the availability of credentials.

	Args:
		credentials_status: The status indicating if credentials are valid.
	"""

	if credentials_status:
		# Show a select box with the years available in the dataset
		available_years = st.session_state.dataframe["Fecha de venta"].dt.year.unique()
		compare_multiple_years: bool = st.checkbox("Compare multiple years.")

		if compare_multiple_years:
			# Select all the years availables
			selected_years = st.multiselect(
				"Select all the years you want to compare",
				available_years,
				default=2023,
			)

			compare_products_years(df=st.session_state.dataframe, years=selected_years)
			compare_income_month_years(
				df=st.session_state.dataframe, years=selected_years
			)
			flow_money(df=st.session_state.dataframe, years=selected_years)
		else:
			selected_year: str = st.selectbox("Select a year", available_years)

			# Number of products sold month/year
			product_month(df=st.session_state.dataframe, year=int(selected_year))

			# Money earned per month/year
			money_month(df=st.session_state.dataframe, year=int(selected_year))

			# Number of people interested by category in a year
			interest_category(df=st.session_state.dataframe, year=int(selected_year))

			col1, col2 = st.columns(2)
			with col1:
				# Matrix confusion relation between product status and gender
				gender_status(df=st.session_state.dataframe, year=int(selected_year))
			with col2:
				# Matrix confusion relation between product state and country
				status_country(df=st.session_state.dataframe, year=int(selected_year))

			# Matrix confusion relation between genre and country
			gender_country(df=st.session_state.dataframe, year=int(selected_year))


# First call to the config page function
config_streamlit_page(page_name="Graphs")

# Check if the credentials are available and display all the graphs
display_all_graphs(credentials_status=check_credentials())
