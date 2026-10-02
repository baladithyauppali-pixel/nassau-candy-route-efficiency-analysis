import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Nassau Candy Route Efficiency",
    page_icon="🍬",
    layout="wide",
)

# -----------------------------
# Page styling
# -----------------------------
st.markdown("""
<style>
    .main-title {font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem;}
    .sub-title {color: #667085; margin-bottom: 1.2rem;}
    .quality-box {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border: 1px solid #F2C94C;
        background: #FFF9E6;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🍬 Nassau Candy — Shipping Route Efficiency</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Factory-to-customer logistics analysis | Business Analyst project</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    candidates = [
        Path("Nassau Candy Distributor.csv"),
        Path("data/Nassau Candy Distributor.csv"),
        Path("/mnt/data/Nassau Candy Distributor.csv"),
    ]
    for p in candidates:
        if p.exists():
            data = pd.read_csv(p)
            return data
    raise FileNotFoundError(
        "Dataset not found. Put 'Nassau Candy Distributor.csv' in the repository root "
        "or inside a 'data' folder."
    )

df = load_data()

# -----------------------------
# Product -> Factory mapping
# -----------------------------
factory_map = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory",
}

factory_coords = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.119140, -96.181150),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.117500, -89.971107),
}

# State centroids used for a lightweight geographic visualization.
state_centroids = {
    "Alabama": (32.8, -86.8), "Alaska": (64.2, -149.5), "Arizona": (34.3, -111.7),
    "Arkansas": (35.0, -92.4), "California": (36.8, -119.4), "Colorado": (39.0, -105.5),
    "Connecticut": (41.6, -72.7), "Delaware": (39.0, -75.5), "Florida": (28.6, -82.4),
    "Georgia": (32.7, -83.3), "Hawaii": (20.8, -156.3), "Idaho": (44.2, -114.4),
    "Illinois": (40.0, -89.2), "Indiana": (39.9, -86.3), "Iowa": (42.0, -93.5),
    "Kansas": (38.5, -98.4), "Kentucky": (37.5, -85.3), "Louisiana": (31.0, -92.0),
    "Maine": (45.3, -69.0), "Maryland": (39.0, -76.7), "Massachusetts": (42.3, -71.8),
    "Michigan": (44.3, -84.5), "Minnesota": (46.3, -94.3), "Mississippi": (32.7, -89.7),
    "Missouri": (38.5, -92.5), "Montana": (47.0, -110.0), "Nebraska": (41.5, -99.8),
    "Nevada": (39.3, -116.6), "New Hampshire": (43.7, -71.6), "New Jersey": (40.2, -74.7),
    "New Mexico": (34.5, -106.0), "New York": (42.9, -75.5), "North Carolina": (35.5, -79.4),
    "North Dakota": (47.5, -100.5), "Ohio": (40.4, -82.8), "Oklahoma": (35.6, -97.5),
    "Oregon": (44.0, -120.5), "Pennsylvania": (41.0, -77.8), "Rhode Island": (41.7, -71.5),
    "South Carolina": (33.8, -80.9), "South Dakota": (44.4, -100.2), "Tennessee": (35.8, -86.4),
    "Texas": (31.5, -99.3), "Utah": (39.3, -111.7), "Vermont": (44.0, -72.7),
    "Virginia": (37.5, -78.8), "Washington": (47.4, -120.5), "West Virginia": (38.6, -80.6),
    "Wisconsin": (44.5, -89.5), "Wyoming": (43.0, -107.6),
    "District of Columbia": (38.9, -77.0),
}

# -----------------------------
# Prepare data
# -----------------------------
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")
df["Shipping Lead Time"] = (df["Ship Date"] - df["Order Date"]).dt.days
df["Factory"] = df["Product Name"].map(factory_map)
df["Route"] = df["Factory"].fillna("Unknown Factory") + " → " + df["State/Province"].astype(str)

# A route is counted at order level, not line-item level.
order_level = (
    df.sort_values(["Order ID", "Order Date"])
      .groupby("Order ID", as_index=False)
      .agg({
          "Order Date": "first",
          "Ship Date": "first",
          "Shipping Lead Time": "first",
          "Ship Mode": "first",
          "Customer ID": "first",
          "State/Province": "first",
          "Region": "first",
          "Sales": "sum",
          "Units": "sum",
          "Gross Profit": "sum",
          "Product Name": "first",
          "Factory": "first",
      })
)
order_level["Route"] = (
    order_level["Factory"].fillna("Unknown Factory")
    + " → "
    + order_level["State/Province"].astype(str)
)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("🔎 Filters")

min_date = df["Order Date"].min()
max_date = df["Order Date"].max()

date_range = st.sidebar.date_input(
    "Order date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_date, end_date = min_date, max_date

regions = sorted(df["Region"].dropna().unique().tolist())
selected_regions = st.sidebar.multiselect("Region", regions, default=regions)

states = sorted(df["State/Province"].dropna().unique().tolist())
selected_states = st.sidebar.multiselect("State / Province", states, default=[])

ship_modes = sorted(df["Ship Mode"].dropna().unique().tolist())
selected_modes = st.sidebar.multiselect("Ship Mode", ship_modes, default=ship_modes)

threshold = st.sidebar.slider(
    "Delay threshold (days)",
    min_value=1,
    max_value=60,
    value=10,
    step=1,
)

# Apply filters at order level.
filtered = order_level[
    (order_level["Order Date"] >= start_date)
    & (order_level["Order Date"] <= end_date)
    & (order_level["Region"].isin(selected_regions))
    & (order_level["Ship Mode"].isin(selected_modes))
].copy()

if selected_states:
    filtered = filtered[filtered["State/Province"].isin(selected_states)].copy()

filtered["Delay Flag"] = filtered["Shipping Lead Time"] > threshold

# -----------------------------
# KPI cards
# -----------------------------
total_orders = filtered["Order ID"].nunique()
total_sales = filtered["Sales"].sum()
total_units = filtered["Units"].sum()
avg_lead = filtered["Shipping Lead Time"].mean()
avg_profit = filtered["Gross Profit"].sum()
route_count = filtered["Route"].nunique()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Orders", f"{total_orders:,}")
c2.metric("Sales", f"${total_sales:,.0f}")
c3.metric("Units", f"{total_units:,.0f}")
c4.metric("Avg Lead Time", f"{avg_lead:,.0f} days" if pd.notna(avg_lead) else "—")
c5.metric("Gross Profit", f"${avg_profit:,.0f}")
c6.metric("Routes", f"{route_count:,}")

# -----------------------------
# Data quality alert
# -----------------------------
raw_min = df["Shipping Lead Time"].min()
raw_max = df["Shipping Lead Time"].max()
raw_median = df["Shipping Lead Time"].median()

if raw_max > 90:
    st.markdown(
        f"""
        <div class="quality-box">
        <b>⚠️ Data Quality Alert</b><br>
        The source dates produce unusually high calculated shipping lead times:
        minimum <b>{raw_min:,.0f}</b> days, median <b>{raw_median:,.0f}</b> days,
        maximum <b>{raw_max:,.0f}</b> days. Validate shipment-date business meaning
        before using these figures as operational SLA measures.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🚚 Route Efficiency",
    "🗺️ Geographic Analysis",
    "🚛 Ship Mode",
    "🔎 Route Drill-Down",
])

with tab1:
    st.subheader("Executive Overview")

    left, right = st.columns(2)

    with left:
        region_summary = (
            filtered.groupby("Region", as_index=False)
            .agg(Orders=("Order ID", "nunique"), Sales=("Sales", "sum"))
            .sort_values("Orders", ascending=False)
        )
        fig = px.bar(
            region_summary,
            x="Region",
            y="Orders",
            title="Orders by Region",
            text_auto=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        mode_summary = (
            filtered.groupby("Ship Mode", as_index=False)
            .agg(Orders=("Order ID", "nunique"), Avg_Lead=("Shipping Lead Time", "mean"))
        )
        fig = px.bar(
            mode_summary,
            x="Ship Mode",
            y="Avg_Lead",
            title="Average Lead Time by Ship Mode",
            text_auto=".1f",
        )
        fig.update_yaxes(title="Average lead time (days)")
        st.plotly_chart(fig, use_container_width=True)

    factory_summary = (
        filtered.groupby("Factory", as_index=False)
        .agg(Orders=("Order ID", "nunique"), Sales=("Sales", "sum"), Avg_Lead=("Shipping Lead Time", "mean"))
        .sort_values("Orders", ascending=False)
    )
    st.subheader("Factory Performance")
    st.dataframe(factory_summary, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Factory → Customer State Route Efficiency")

    route_summary = (
        filtered.groupby("Route", as_index=False)
        .agg(
            Orders=("Order ID", "nunique"),
            Avg_Lead_Time=("Shipping Lead Time", "mean"),
            Lead_Time_Std=("Shipping Lead Time", "std"),
            Delay_Frequency=("Delay Flag", "mean"),
            Sales=("Sales", "sum"),
        )
    )
    route_summary["Lead_Time_Std"] = route_summary["Lead_Time_Std"].fillna(0)
    route_summary["Delay_Frequency"] *= 100
    if not route_summary.empty:
        fastest = route_summary["Avg_Lead_Time"].min()
        route_summary["Efficiency Score"] = (
            100 * fastest / route_summary["Avg_Lead_Time"].replace(0, np.nan)
        ).round(1)

    st.markdown("**Route performance leaderboard**")
    display_routes = route_summary.sort_values(
        ["Avg_Lead_Time", "Orders"], ascending=[True, False]
    )
    st.dataframe(display_routes, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**10 routes with lowest average lead time**")
        st.dataframe(
            route_summary.sort_values("Avg_Lead_Time").head(10),
            use_container_width=True,
            hide_index=True,
        )
    with col_b:
        st.markdown("**10 routes with highest average lead time**")
        st.dataframe(
            route_summary.sort_values("Avg_Lead_Time", ascending=False).head(10),
            use_container_width=True,
            hide_index=True,
        )

    if not route_summary.empty:
        fig = px.scatter(
            route_summary,
            x="Orders",
            y="Avg_Lead_Time",
            size="Sales",
            hover_name="Route",
            title="Route Volume vs Average Lead Time",
        )
        fig.update_yaxes(title="Average lead time (days)")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Geographic Bottleneck Analysis")

    state_summary = (
        filtered.groupby(["State/Province", "Region"], as_index=False)
        .agg(
            Orders=("Order ID", "nunique"),
            Avg_Lead_Time=("Shipping Lead Time", "mean"),
            Delay_Frequency=("Delay Flag", "mean"),
            Sales=("Sales", "sum"),
        )
    )
    state_summary["Delay_Frequency"] *= 100

    st.dataframe(
        state_summary.sort_values("Avg_Lead_Time", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    map_df = state_summary.copy()
    map_df["lat"] = map_df["State/Province"].map(lambda x: state_centroids.get(x, (np.nan, np.nan))[0])
    map_df["lon"] = map_df["State/Province"].map(lambda x: state_centroids.get(x, (np.nan, np.nan))[1])
    map_df = map_df.dropna(subset=["lat", "lon"])

    if not map_df.empty:
        fig = px.scatter_geo(
            map_df,
            lat="lat",
            lon="lon",
            size="Orders",
            color="Avg_Lead_Time",
            hover_name="State/Province",
            hover_data=["Region", "Orders", "Avg_Lead_Time", "Delay_Frequency"],
            scope="usa",
            title="State-Level Shipping Efficiency",
            color_continuous_scale="Turbo",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### High-volume / poor-performance states")
    if not state_summary.empty:
        volume_cutoff = state_summary["Orders"].median()
        lead_cutoff = state_summary["Avg_Lead_Time"].median()
        bottlenecks = state_summary[
            (state_summary["Orders"] >= volume_cutoff)
            & (state_summary["Avg_Lead_Time"] >= lead_cutoff)
        ].sort_values(["Avg_Lead_Time", "Orders"], ascending=[False, False])
        st.dataframe(bottlenecks, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Ship Mode Performance")

    mode_summary = (
        filtered.groupby("Ship Mode", as_index=False)
        .agg(
            Orders=("Order ID", "nunique"),
            Avg_Lead_Time=("Shipping Lead Time", "mean"),
            Lead_Time_Std=("Shipping Lead Time", "std"),
            Delay_Frequency=("Delay Flag", "mean"),
            Sales=("Sales", "sum"),
            Gross_Profit=("Gross Profit", "sum"),
        )
    )
    mode_summary["Lead_Time_Std"] = mode_summary["Lead_Time_Std"].fillna(0)
    mode_summary["Delay_Frequency"] *= 100

    st.dataframe(mode_summary, use_container_width=True, hide_index=True)

    fig = px.bar(
        mode_summary,
        x="Ship Mode",
        y="Avg_Lead_Time",
        title="Average Lead Time by Shipping Method",
        text_auto=".1f",
    )
    fig.update_yaxes(title="Average lead time (days)")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Cost-time tradeoffs should be interpreted descriptively from this dataset. "
        "The dataset does not provide a direct shipping-charge field, so shipping cost "
        "cannot be quantified from these records alone."
    )

with tab5:
    st.subheader("Route Drill-Down")

    route_options = sorted(filtered["Route"].dropna().unique().tolist())
    selected_route = st.selectbox("Select a route", route_options)

    if selected_route:
        route_orders = filtered[filtered["Route"] == selected_route].copy()
        st.write(f"**{selected_route}** — {len(route_orders):,} orders")

        drill_cols = [
            "Order ID", "Order Date", "Ship Date", "Shipping Lead Time",
            "Ship Mode", "Customer ID", "State/Province", "Region",
            "Sales", "Units", "Gross Profit"
        ]
        st.dataframe(
            route_orders[drill_cols].sort_values("Order Date"),
            use_container_width=True,
            hide_index=True,
        )

        timeline = route_orders.sort_values("Order Date")
        if not timeline.empty:
            fig = px.scatter(
                timeline,
                x="Order Date",
                y="Shipping Lead Time",
                color="Ship Mode",
                hover_data=["Order ID", "State/Province", "Sales"],
                title="Order-Level Shipment Timeline",
            )
            fig.update_yaxes(title="Lead time (days)")
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "Nassau Candy Distributor — Business Analyst Internship Project | "
    "Lead-time results should be validated against source-system shipment-date definitions."
)
