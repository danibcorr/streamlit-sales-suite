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
	Renders a bar chart showing total income per month for
	a given year, with all 12 months displayed and
	zero-filled where no sales occurred.

	Args:
		df: The DataFrame containing sales data.
		year: The year to filter and visualize.

	Returns:
		None.
	"""

	st.subheader(f"Money Earned per Month in {year}")

	yearly_sales = df[df["Fecha de venta"].dt.year == year].copy()
	yearly_sales["Mes"] = yearly_sales["Fecha de venta"].dt.month

	income_by_month = yearly_sales.groupby("Mes")["Precio producto"].sum()
	income_by_month = income_by_month.reindex(range(1, 13), fill_value=0)

	monthly_income_df = pd.DataFrame(
		{"Mes": MONTHS_NAMES, "Dinero": income_by_month.values}
	)
	monthly_income_df["Mes"] = pd.Categorical(
		monthly_income_df["Mes"], categories=MONTHS_NAMES, ordered=True
	)

	fig = px.bar(
		monthly_income_df,
		x="Mes",
		y="Dinero",
		labels={"Dinero": "Income (€)", "Mes": "Month"},
		color="Mes",
		color_discrete_sequence=px.colors.qualitative.Pastel,
	)
	fig.update_layout(
		xaxis_title="Month",
		yaxis_title="Income (€)",
		xaxis_tickangle=-45,
		showlegend=False,
		coloraxis_showscale=False,
	)
	fig.update_traces(texttemplate="%{y:.0f}€", textposition="outside")

	st.plotly_chart(fig, width="stretch")


def product_month(df: pd.DataFrame, year: int) -> None:
	"""
	Renders a stacked horizontal bar chart of products sold
	per month by type, alongside KPI metrics including the
	top 3 product types, total products sold, and total
	revenue for the given year.

	Args:
		df: The DataFrame containing sales data.
		year: The year to filter and visualize.

	Returns:
		None.
	"""

	st.subheader(f"Products Sold per Month in {year}")

	chart_col, metrics_col = st.columns(2, gap="large")

	yearly_sales = df[df["Fecha de venta"].dt.year == year].copy()
	products_by_month = yearly_sales.copy()
	products_by_month["Mes"] = yearly_sales["Fecha de venta"].dt.month
	products_by_month = (
		products_by_month.groupby(["Mes", "Tipo producto"])
		.size()
		.reset_index(name="count")
	)

	products_by_month["Mes"] = products_by_month["Mes"].apply(
		lambda x: MONTHS_NAMES[x - 1]
	)
	products_by_month["Mes"] = pd.Categorical(
		products_by_month["Mes"], categories=MONTHS_NAMES, ordered=True
	)

	pivoted_products = products_by_month.pivot(
		index="Mes", columns="Tipo producto", values="count"
	).fillna(0)
	pivoted_products = pivoted_products.loc[(pivoted_products > 0).any(axis=1)]

	with chart_col:
		fig = go.Figure()

		for product_type in pivoted_products.columns:
			fig.add_trace(
				go.Bar(
					y=pivoted_products.index,
					x=pivoted_products[product_type],
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

		st.plotly_chart(fig, width="stretch")

	with metrics_col:
		top_3_products = obtain_top(df=yearly_sales, top=3, column="Tipo producto")
		st.metric(
			label="Top 3 Products", value=str(", ".join(top_3_products)), border=True
		)

		total_col, revenue_col = st.columns(2)
		total_col.metric(
			label="Total Products Sold",
			value=int(yearly_sales["Tipo producto"].value_counts().sum()),
			border=True,
		)
		revenue_col.metric(
			label="Total Cash Obtained",
			value=str(float(yearly_sales["Precio producto"].sum().round(4))) + " €",
			border=True,
		)


def gender_status(df: pd.DataFrame, year: int) -> None:
	"""
	Renders a heatmap showing the relationship between
	buyer gender and product status for a given year.

	Args:
		df: The DataFrame containing sales data.
		year: The year to filter and visualize.

	Returns:
		None.
	"""

	st.subheader(f"Gender Status Heatmap in {year}")

	yearly_sales = df[df["Fecha de venta"].dt.year == year].copy()

	gender_status_pivot = yearly_sales.pivot_table(
		index="Estado del producto",
		columns="Genero",
		values="Fecha de venta",
		aggfunc="count",
		fill_value=0,
	)

	fig = px.imshow(
		gender_status_pivot,
		text_auto=True,
		color_continuous_scale="mint",
		labels={"color": "Count"},
		aspect="auto",
	)
	fig.update_layout(
		xaxis_title="Gender", yaxis_title="Product Status", coloraxis_showscale=False
	)

	st.plotly_chart(fig, width="stretch")


def status_country(df: pd.DataFrame, year: int) -> None:
	"""
	Renders a heatmap showing the distribution of product
	statuses across countries for a given year.

	Args:
		df: The DataFrame containing sales data.
		year: The year to filter and visualize.

	Returns:
		None.
	"""

	st.subheader(f"Product Status by Country Heatmap in {year}")

	yearly_sales = df[df["Fecha de venta"].dt.year == year].copy()

	status_country_pivot = yearly_sales.pivot_table(
		index="Pais",
		columns="Estado del producto",
		values="Fecha de venta",
		aggfunc="count",
		fill_value=0,
	)

	fig = px.imshow(
		status_country_pivot,
		text_auto=True,
		color_continuous_scale="mint",
		labels={"color": "Count"},
		aspect="auto",
	)
	fig.update_layout(
		xaxis_title="Product Status",
		yaxis_title="Country",
		coloraxis_showscale=False,
	)

	st.plotly_chart(fig, width="stretch")


def gender_country(df: pd.DataFrame, year: int) -> None:
	"""
	Renders side-by-side heatmaps showing the distribution
	of sales by country, split by gender, for a given year.

	Args:
		df: The DataFrame containing sales data.
		year: The year to filter and visualize.

	Returns:
		None.
	"""

	st.subheader(f"Gender Distribution by Country in {year}")

	yearly_sales = df[df["Fecha de venta"].dt.year == year].copy()

	gender_country_pivot = yearly_sales.pivot_table(
		index="Pais",
		columns="Genero",
		values="Fecha de venta",
		aggfunc="count",
		fill_value=0,
	)

	female_by_country = (
		gender_country_pivot["F"].sort_values(ascending=False).to_frame()
	)
	male_by_country = gender_country_pivot["M"].sort_values(ascending=False).to_frame()

	female_col, male_col = st.columns(2)
	with female_col:
		fig = px.imshow(
			female_by_country,
			text_auto=True,
			color_continuous_scale="mint",
			labels={"color": "Count"},
			aspect="auto",
		)
		fig.update_layout(
			xaxis_title="Gender",
			yaxis_title="Country",
			coloraxis_showscale=False,
		)
		st.plotly_chart(fig, width="stretch")
	with male_col:
		fig = px.imshow(
			male_by_country,
			text_auto=True,
			color_continuous_scale="mint",
			labels={"color": "Count"},
			aspect="auto",
		)
		fig.update_layout(
			xaxis_title="Gender",
			yaxis_title="Country",
			coloraxis_showscale=False,
		)
		st.plotly_chart(fig, width="stretch")


def compare_products_years(df: pd.DataFrame, years: list[int]) -> None:
	"""
	Renders a grouped bar chart comparing the quantity of
	products sold by type across multiple years.

	Args:
		df: The DataFrame containing sales data.
		years: The years to compare.

	Returns:
		None.
	"""

	st.subheader(
		"Comparison of Products Sold by Type and Year in "
		f"{str(', '.join([str(year) for year in years]))}"
	)

	if len(years) != 0:
		filtered_by_years = df[df["Fecha de venta"].dt.year.isin(years)]

		products_by_year = (
			filtered_by_years.groupby(
				["Tipo producto", filtered_by_years["Fecha de venta"].dt.year]
			)
			.size()
			.reset_index(name="Cantidad")
		)
		products_by_year.columns = ["Tipo producto", "Año", "Cantidad"]
		products_by_year["Año"] = products_by_year["Año"].astype(str)

		fig = px.bar(
			products_by_year,
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

		st.plotly_chart(fig, width="stretch")

	else:
		st.error("No years selected", icon="⚠️")


def compare_income_month_years(df: pd.DataFrame, years: list[int]) -> None:
	"""
	Renders a grouped bar chart comparing monthly income
	across multiple years.

	Args:
		df: The DataFrame containing sales data.
		years: The years to compare.

	Returns:
		None.
	"""

	st.subheader(
		"Comparison of Income by Month and Year in "
		f"{str(', '.join([str(year) for year in years]))}"
	)

	if len(years) != 0:
		filtered_by_years = df[df["Fecha de venta"].dt.year.isin(years)].copy()
		filtered_by_years.loc[:, "Año"] = filtered_by_years[
			"Fecha de venta"
		].dt.year.astype(str)
		filtered_by_years.loc[:, "Mes"] = filtered_by_years["Fecha de venta"].dt.month
		filtered_by_years.loc[:, "Mes_nombre"] = filtered_by_years[
			"Fecha de venta"
		].dt.strftime("%B")
		filtered_by_years.loc[:, "Año_Mes"] = filtered_by_years[
			"Fecha de venta"
		].dt.strftime("%Y-%m")

		income_by_month_year = (
			filtered_by_years.groupby(["Año", "Mes", "Mes_nombre", "Año_Mes"])[
				"Precio producto"
			]
			.sum()
			.reset_index()
		)

		fig = px.bar(
			income_by_month_year,
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

		st.plotly_chart(fig, width="stretch")

	else:
		st.error("No years selected", icon="⚠️")


def flow_money(df: pd.DataFrame, years: list[int]) -> None:
	"""
	Displays aggregated financial metrics and a detailed
	annual breakdown table for the selected years.

	Args:
		df: The DataFrame containing sales data.
		years: The years to compare.

	Returns:
		None.
	"""

	st.subheader(
		f"Financial Flow Analysis: {str(', '.join([str(year) for year in years]))}"
	)

	if len(years) != 0:
		all_monthly_summaries, all_annual_summaries = [], []

		for year in years:
			monthly, annual = summarize_year(df, year)
			all_monthly_summaries.append(monthly)
			all_annual_summaries.append(annual)

		annual_overview_df = pd.DataFrame(all_annual_summaries)

		revenue_col, products_col, _ = st.columns(3)
		revenue_col.metric(
			"Total Revenue", f"{annual_overview_df['Total_Revenue'].sum():,.2f} €"
		)
		products_col.metric(
			"Total Products", f"{annual_overview_df['Total_Products'].sum():,}"
		)

		st.dataframe(annual_overview_df.round(2), width="stretch")

	else:
		st.error("No years selected", icon="⚠️")


def category_sales_by_period(df: pd.DataFrame, years: list[int]) -> None:
	"""
	Displays product categories sold per month with quantity
	and total price for the given year(s).

	Args:
		df: The DataFrame containing sales data.
		years: The year(s) to filter and visualize.

	Returns:
		None.
	"""

	st.subheader(
		f"Category Sales by Month in {str(', '.join([str(year) for year in years]))}"
	)

	if len(years) != 0:
		filtered_sales = df[df["Fecha de venta"].dt.year.isin(years)].copy()
		filtered_sales["Año"] = filtered_sales["Fecha de venta"].dt.year
		filtered_sales["Mes"] = filtered_sales["Fecha de venta"].dt.month

		all_months = sorted(filtered_sales["Mes"].unique())
		month_names = [MONTHS_NAMES[m - 1] for m in all_months]

		selected_month_name = st.selectbox("Select a month", month_names)
		selected_month = all_months[month_names.index(selected_month_name)]

		month_data = filtered_sales[filtered_sales["Mes"] == selected_month]

		summary = (
			month_data.groupby(["Tipo producto", "Año"])
			.agg(
				Quantity=("Tipo producto", "count"),
				Total_Price=("Precio producto", "sum"),
			)
			.reset_index()
		)
		summary["Año"] = summary["Año"].astype(str)

		col1, col2 = st.columns(2)

		with col1:
			fig = px.bar(
				summary,
				x="Tipo producto",
				y="Quantity",
				color="Año",
				barmode="group",
				title="Quantity by Category",
				color_discrete_sequence=px.colors.qualitative.Pastel,
			)
			fig.update_layout(xaxis_tickangle=-45)
			st.plotly_chart(fig, width="stretch")

		with col2:
			fig = px.bar(
				summary,
				x="Tipo producto",
				y="Total_Price",
				color="Año",
				barmode="group",
				title="Total Price (€) by Category",
				color_discrete_sequence=px.colors.qualitative.Pastel,
			)
			fig.update_layout(xaxis_tickangle=-45)
			st.plotly_chart(fig, width="stretch")

	else:
		st.error("No years selected", icon="⚠️")


def display_all_graphs(has_valid_credentials: bool) -> None:
	"""
	Orchestrates the rendering of all graph pages based on
	user-selected year(s). In single-year mode, displays
	per-month breakdowns and heatmaps. In multi-year mode,
	displays comparative charts and financial flow analysis.

	Args:
		has_valid_credentials: Whether valid credentials are
			present to access the data.

	Returns:
		None.
	"""

	if has_valid_credentials:
		available_years = st.session_state.dataframe["Fecha de venta"].dt.year.unique()
		is_multi_year_comparison: bool = st.checkbox("Compare multiple years.")

		if is_multi_year_comparison:
			selected_years = st.multiselect(
				"Select all the years you want to compare",
				available_years,
			)

			compare_products_years(df=st.session_state.dataframe, years=selected_years)
			compare_income_month_years(
				df=st.session_state.dataframe, years=selected_years
			)
			flow_money(df=st.session_state.dataframe, years=selected_years)
			category_sales_by_period(
				df=st.session_state.dataframe, years=selected_years
			)
		else:
			selected_year: str = st.selectbox("Select a year", available_years)

			product_month(df=st.session_state.dataframe, year=int(selected_year))
			money_month(df=st.session_state.dataframe, year=int(selected_year))

			heatmap_left_col, heatmap_right_col = st.columns(2)
			with heatmap_left_col:
				gender_status(df=st.session_state.dataframe, year=int(selected_year))
			with heatmap_right_col:
				status_country(df=st.session_state.dataframe, year=int(selected_year))

			gender_country(df=st.session_state.dataframe, year=int(selected_year))
			category_sales_by_period(
				df=st.session_state.dataframe, years=[int(selected_year)]
			)


config_streamlit_page(page_name="Graphs")
display_all_graphs(has_valid_credentials=check_credentials())
