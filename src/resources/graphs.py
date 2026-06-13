# 3pps
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

# Own modules
from config import MONTHS_NAMES
from config.constants import (
    COL_COUNT,
    COL_ESTADO_PRODUCTO,
    COL_FECHA_VENTA,
    COL_GENERO,
    COL_INCOME,
    COL_MONTH,
    COL_MONTH_NAME,
    COL_MONTH_NUM,
    COL_PAIS,
    COL_PRECIO_PRODUCTO,
    COL_QUANTITY,
    COL_TIPO_PRODUCTO,
    COL_TOTAL_PRICE,
    COL_TOTAL_PRODUCTS,
    COL_TOTAL_REVENUE,
    COL_YEAR,
    COL_YEAR_MONTH,
    COL_YEAR_STR,
    SESSION_DATAFRAME,
)
from utils import (
    check_credentials,
    config_streamlit_page,
    filter_by_year,
    obtain_top,
    summarize_year,
)


def money_month(df: pl.DataFrame, year: int) -> None:
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

	yearly_sales = filter_by_year(df, year).with_columns(
		pl.col(COL_FECHA_VENTA).dt.month().alias(COL_MONTH_NUM)
	)

	income_by_month = yearly_sales.group_by(COL_MONTH_NUM).agg(
		pl.col(COL_PRECIO_PRODUCTO).sum().alias(COL_INCOME)
	)

	all_months = pl.DataFrame({COL_MONTH_NUM: list(range(1, 13))}).cast(
		{COL_MONTH_NUM: income_by_month.schema[COL_MONTH_NUM]}
	)

	monthly_income_df = (
		all_months.join(income_by_month, on=COL_MONTH_NUM, how="left")
		.fill_null(0)
		.sort(COL_MONTH_NUM)
		.with_columns(
			pl.col(COL_MONTH_NUM)
			.map_elements(lambda m: MONTHS_NAMES[m - 1], return_dtype=pl.Utf8)
			.alias(COL_MONTH_NAME)
		)
	)

	plot_df = monthly_income_df.to_pandas()
	plot_df[COL_MONTH_NAME] = pl.Series(MONTHS_NAMES).to_pandas().astype("category")

	fig = px.bar(
		plot_df,
		x=COL_MONTH_NAME,
		y=COL_INCOME,
		labels={COL_INCOME: "Income (€)", COL_MONTH_NAME: COL_MONTH},
		color=COL_MONTH_NAME,
		color_discrete_sequence=px.colors.qualitative.Pastel,
	)
	fig.update_layout(
		xaxis_title=COL_MONTH,
		yaxis_title="Income (€)",
		xaxis_tickangle=-45,
		showlegend=False,
		coloraxis_showscale=False,
	)
	fig.update_traces(texttemplate="%{y:.0f}€", textposition="outside")

	st.plotly_chart(fig, width="stretch")


def product_month(df: pl.DataFrame, year: int) -> None:
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

	yearly_sales = filter_by_year(df, year)

	products_by_month = (
		yearly_sales.with_columns(
			pl.col(COL_FECHA_VENTA).dt.month().alias(COL_MONTH_NUM)
		)
		.group_by([COL_MONTH_NUM, COL_TIPO_PRODUCTO])
		.len()
		.rename({"len": COL_COUNT})
		.with_columns(
			pl.col(COL_MONTH_NUM)
			.map_elements(lambda m: MONTHS_NAMES[m - 1], return_dtype=pl.Utf8)
			.alias(COL_MONTH_NAME)
		)
	)

	pivoted = products_by_month.pivot(
		on=COL_TIPO_PRODUCTO,
		index=COL_MONTH_NAME,
		values=COL_COUNT,
	).fill_null(0)

	# Sort by month order
	month_order = {name: i for i, name in enumerate(MONTHS_NAMES)}
	pivoted_pd = pivoted.to_pandas().set_index(COL_MONTH_NAME)
	pivoted_pd = pivoted_pd.sort_index(
		key=lambda idx: idx.map(lambda x: month_order.get(x, 99))
	)
	pivoted_pd = pivoted_pd.loc[(pivoted_pd > 0).any(axis=1)]

	with chart_col:
		fig = go.Figure()

		for product_type in pivoted_pd.columns:
			fig.add_trace(
				go.Bar(
					y=pivoted_pd.index,
					x=pivoted_pd[product_type],
					name=product_type,
					orientation="h",
				)
			)

		fig.update_layout(
			xaxis_title="Number of Products",
			yaxis_title=COL_MONTH,
			barmode="stack",
			height=400,
			legend=dict(
				orientation="h",
				yanchor="bottom",
				y=1.02,
				xanchor="right",
				x=1,
			),
		)

		st.plotly_chart(fig, width="stretch")

	with metrics_col:
		top_3_products = obtain_top(df=yearly_sales, top=3, column=COL_TIPO_PRODUCTO)
		st.metric(
			label="Top 3 Products",
			value=", ".join(top_3_products),
			border=True,
		)

		total_col, revenue_col = st.columns(2)
		total_col.metric(
			label="Total Products Sold",
			value=yearly_sales.height,
			border=True,
		)
		revenue_col.metric(
			label="Total Cash Obtained",
			value=f"{yearly_sales.get_column(COL_PRECIO_PRODUCTO).sum():.4f} €",
			border=True,
		)


def gender_status(df: pl.DataFrame, year: int) -> None:
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

	yearly_sales = filter_by_year(df, year)

	pivot_pd = (
		yearly_sales.with_columns(
			pl.col(COL_GENERO).fill_null("Unknown"),
			pl.col(COL_ESTADO_PRODUCTO).fill_null("Unknown"),
		)
		.group_by([COL_ESTADO_PRODUCTO, COL_GENERO])
		.len()
		.pivot(on=COL_GENERO, index=COL_ESTADO_PRODUCTO, values="len")
		.fill_null(0)
		.to_pandas()
		.set_index(COL_ESTADO_PRODUCTO)
	)

	fig = px.imshow(
		pivot_pd,
		text_auto=True,
		color_continuous_scale="mint",
		labels={"color": "Count"},
		aspect="auto",
	)
	fig.update_layout(
		xaxis_title="Gender",
		yaxis_title="Product Status",
		coloraxis_showscale=False,
	)

	st.plotly_chart(fig, width="stretch")


def status_country(df: pl.DataFrame, year: int) -> None:
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

	yearly_sales = filter_by_year(df, year)

	pivot_pd = (
		yearly_sales.group_by([COL_PAIS, COL_ESTADO_PRODUCTO])
		.len()
		.pivot(on=COL_ESTADO_PRODUCTO, index=COL_PAIS, values="len")
		.fill_null(0)
		.to_pandas()
		.set_index(COL_PAIS)
	)

	fig = px.imshow(
		pivot_pd,
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


def gender_country(df: pl.DataFrame, year: int) -> None:
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

	yearly_sales = filter_by_year(df, year)

	pivot_pd = (
		yearly_sales.group_by([COL_PAIS, COL_GENERO])
		.len()
		.pivot(on=COL_GENERO, index=COL_PAIS, values="len")
		.fill_null(0)
		.to_pandas()
		.set_index(COL_PAIS)
	)

	female_by_country = pivot_pd[["F"]].sort_values("F", ascending=False)
	male_by_country = pivot_pd[["M"]].sort_values("M", ascending=False)

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


def compare_products_years(df: pl.DataFrame, years: list[int]) -> None:
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
		f"{', '.join([str(y) for y in years])}"
	)

	if not years:
		st.error("No years selected", icon="⚠️")
		return

	filtered = df.filter(pl.col(COL_FECHA_VENTA).dt.year().is_in(years))

	products_by_year = (
		filtered.with_columns(
			pl.col(COL_FECHA_VENTA).dt.year().cast(pl.Utf8).alias(COL_YEAR_STR)
		)
		.group_by([COL_TIPO_PRODUCTO, COL_YEAR_STR])
		.len()
		.rename({"len": COL_QUANTITY})
	)

	fig = px.bar(
		products_by_year.to_pandas(),
		x=COL_TIPO_PRODUCTO,
		y=COL_QUANTITY,
		color=COL_YEAR_STR,
		title="",
		barmode="group",
		text_auto=True,
		color_discrete_sequence=px.colors.qualitative.Pastel,
	)
	fig.update_layout(
		xaxis_title="Product Type",
		yaxis_title=COL_QUANTITY,
		legend_title=COL_YEAR,
		barmode="group",
		xaxis_tickangle=-45,
	)
	fig.update_traces(
		textfont_size=12,
		textangle=0,
		textposition="outside",
		cliponaxis=False,
	)

	st.plotly_chart(fig, width="stretch")


def compare_income_month_years(df: pl.DataFrame, years: list[int]) -> None:
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
		f"{', '.join([str(y) for y in years])}"
	)

	if not years:
		st.error("No years selected", icon="⚠️")
		return

	filtered = df.filter(pl.col(COL_FECHA_VENTA).dt.year().is_in(years)).with_columns(
		pl.col(COL_FECHA_VENTA).dt.year().cast(pl.Utf8).alias(COL_YEAR_STR),
		pl.col(COL_FECHA_VENTA).dt.month().alias(COL_MONTH_NUM),
		pl.col(COL_FECHA_VENTA).dt.strftime("%B").alias(COL_MONTH_NAME),
		pl.col(COL_FECHA_VENTA).dt.strftime("%Y-%m").alias(COL_YEAR_MONTH),
	)

	income_by_month_year = (
		filtered.group_by([COL_YEAR_STR, COL_MONTH_NUM, COL_MONTH_NAME, COL_YEAR_MONTH])
		.agg(pl.col(COL_PRECIO_PRODUCTO).sum())
		.sort([COL_YEAR_STR, COL_MONTH_NUM])
	)

	fig = px.bar(
		income_by_month_year.to_pandas(),
		x=COL_MONTH_NAME,
		y=COL_PRECIO_PRODUCTO,
		color=COL_YEAR_STR,
		barmode="group",
		text=COL_PRECIO_PRODUCTO,
		color_discrete_sequence=px.colors.qualitative.Pastel,
		category_orders={
			COL_MONTH_NAME: list(MONTHS_NAMES),
		},
	)
	fig.update_layout(
		xaxis_title=COL_MONTH,
		yaxis_title="Income (€)",
		legend_title=COL_YEAR,
		xaxis_tickangle=-45,
	)
	fig.update_traces(
		texttemplate="%{text:.0f}€",
		textposition="outside",
		textfont_size=10,
		cliponaxis=False,
	)

	st.plotly_chart(fig, width="stretch")


def flow_money(df: pl.DataFrame, years: list[int]) -> None:
	"""
	Displays aggregated financial metrics and a detailed
	annual breakdown table for the selected years.

	Args:
		df: The DataFrame containing sales data.
		years: The years to compare.

	Returns:
		None.
	"""

	st.subheader(f"Financial Flow Analysis: {', '.join([str(y) for y in years])}")

	if not years:
		st.error("No years selected", icon="⚠️")
		return

	all_annual_summaries = []

	for year in years:
		_, annual = summarize_year(df, year)
		all_annual_summaries.append(annual)

	annual_overview_df = pl.DataFrame(all_annual_summaries)

	revenue_col, products_col, _ = st.columns(3)
	total_revenue = annual_overview_df.get_column(COL_TOTAL_REVENUE).sum()
	total_products = annual_overview_df.get_column(COL_TOTAL_PRODUCTS).sum()
	revenue_col.metric("Total Revenue", f"{total_revenue:,.2f} €")
	products_col.metric("Total Products", f"{total_products:,}")

	st.dataframe(annual_overview_df.to_pandas().round(2), width="stretch")


def category_sales_by_period(df: pl.DataFrame, years: list[int]) -> None:
	"""
	Displays product categories sold per month with quantity
	and total price for the given year(s).

	Args:
		df: The DataFrame containing sales data.
		years: The year(s) to filter and visualize.

	Returns:
		None.
	"""

	st.subheader(f"Category Sales by Month in {', '.join([str(y) for y in years])}")

	if not years:
		st.error("No years selected", icon="⚠️")
		return

	filtered_sales = df.filter(
		pl.col(COL_FECHA_VENTA).dt.year().is_in(years)
	).with_columns(
		pl.col(COL_FECHA_VENTA).dt.year().alias(COL_YEAR_STR),
		pl.col(COL_FECHA_VENTA).dt.month().alias(COL_MONTH_NUM),
	)

	all_months = sorted(filtered_sales.get_column(COL_MONTH_NUM).unique().to_list())
	month_names = [MONTHS_NAMES[m - 1] for m in all_months]

	selected_month_name = st.selectbox("Select a month", month_names)
	selected_month = all_months[month_names.index(selected_month_name)]

	month_data = filtered_sales.filter(pl.col(COL_MONTH_NUM) == selected_month)

	summary = (
		month_data.group_by([COL_TIPO_PRODUCTO, COL_YEAR_STR])
		.agg(
			pl.len().alias(COL_QUANTITY),
			pl.col(COL_PRECIO_PRODUCTO).sum().alias(COL_TOTAL_PRICE),
		)
		.with_columns(pl.col(COL_YEAR_STR).cast(pl.Utf8).alias(COL_YEAR_STR))
	)

	summary_pd = summary.to_pandas()

	col1, col2 = st.columns(2)

	with col1:
		fig = px.bar(
			summary_pd,
			x=COL_TIPO_PRODUCTO,
			y=COL_QUANTITY,
			color=COL_YEAR_STR,
			barmode="group",
			title="Quantity by Category",
			color_discrete_sequence=px.colors.qualitative.Pastel,
		)
		fig.update_layout(xaxis_tickangle=-45)
		st.plotly_chart(fig, width="stretch")

	with col2:
		fig = px.bar(
			summary_pd,
			x=COL_TIPO_PRODUCTO,
			y=COL_TOTAL_PRICE,
			color=COL_YEAR_STR,
			barmode="group",
			title="Total Price (€) by Category",
			color_discrete_sequence=px.colors.qualitative.Pastel,
		)
		fig.update_layout(xaxis_tickangle=-45)
		st.plotly_chart(fig, width="stretch")


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
		df: pl.DataFrame | None = st.session_state[SESSION_DATAFRAME]
		if df is None:
			st.warning("No data loaded yet.", icon="⚠️")
			return

		available_years = sorted(
			df.get_column(COL_FECHA_VENTA).dt.year().unique().to_list()
		)
		is_multi_year_comparison: bool = st.checkbox("Compare multiple years.")

		if is_multi_year_comparison:
			selected_years = st.multiselect(
				"Select all the years you want to compare",
				available_years,
			)

			compare_products_years(df=df, years=selected_years)
			compare_income_month_years(df=df, years=selected_years)
			flow_money(df=df, years=selected_years)
			category_sales_by_period(df=df, years=selected_years)
		else:
			selected_year: int = st.selectbox("Select a year", available_years)

			product_month(df=df, year=selected_year)
			money_month(df=df, year=selected_year)

			heatmap_left_col, heatmap_right_col = st.columns(2)
			with heatmap_left_col:
				gender_status(df=df, year=selected_year)
			with heatmap_right_col:
				status_country(df=df, year=selected_year)

			gender_country(df=df, year=selected_year)
			category_sales_by_period(df=df, years=[selected_year])


config_streamlit_page(page_name="Graphs")
display_all_graphs(has_valid_credentials=check_credentials())
