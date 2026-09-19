"""
India Weather & Rainfall Intelligence Dashboard
=================================================
A production-ready Streamlit dashboard for exploring 10 years (2015-2024) of
daily weather and rainfall records across 406 stations / 314 districts / 32
states & UTs in India.

Run with:
    streamlit run app.py

Expects `india_weather_rainfall_data2.csv` in the same folder. If it isn't
found, the app falls back to a file-uploader so it still runs anywhere.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLE
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Weather & Rainfall Intelligence",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Consistent colour theme reused across every chart in the app
PALETTE = {
    "primary": "#0B5FA5",     # deep sky blue   -> rainfall
    "secondary": "#F2994A",   # warm orange     -> temperature
    "accent": "#27AE60",      # green           -> positive/insight highlights
    "muted": "#8395A7",       # slate grey      -> secondary series
    "danger": "#EB5757",      # red             -> extremes/anomalies
}
SEQUENTIAL_BLUES = "Blues"
SEQUENTIAL_ORANGES = "Oranges"
DIVERGING = "RdBu_r"
CATEGORY_SEQUENCE = px.colors.qualitative.Set2
PLOTLY_TEMPLATE = "plotly_white"

MONTH_ORDER = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

STATE_NAME_MAP = {
    "JK": "Jammu & Kashmir", "PB": "Punjab", "HP": "Himachal Pradesh",
    "HR": "Haryana", "CH": "Chandigarh", "UP": "Uttar Pradesh",
    "RJ": "Rajasthan", "DL": "Delhi", "AR": "Arunachal Pradesh",
    "WB": "West Bengal", "SK": "Sikkim", "AS": "Assam",
    "MP": "Madhya Pradesh", "BR": "Bihar", "ML": "Meghalaya",
    "NL": "Nagaland", "GJ": "Gujarat", "TR": "Tripura",
    "MN": "Manipur", "MZ": "Mizoram", "OR": "Odisha",
    "MH": "Maharashtra", "CT": "Chhattisgarh", "DD": "Daman & Diu",
    "KA": "Karnataka", "AP": "Andhra Pradesh", "GA": "Goa",
    "TN": "Tamil Nadu", "LD": "Lakshadweep", "AN": "Andaman & Nicobar Islands",
    "KL": "Kerala", "PY": "Puducherry",
}

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.5rem;}
        
        /* 1. KPIs: Removed white background, added subtle border for dark mode */
        div[data-testid="stMetric"] {
            background-color: transparent;
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 10px;
            padding: 12px 14px;
        }
        div[data-testid="stMetricLabel"] {font-weight: 600; opacity: 0.75;}
        
        /* 2. Insights: Changed to a transparent blue to look great on dark backgrounds */
        .insight-box {
            background-color: rgba(11, 95, 165, 0.15);
            border-left: 4px solid #0B5FA5;
            padding: 10px 16px;
            border-radius: 6px;
            margin: 6px 0 16px 0;
            font-size: 0.92rem;
        }

        /* 3. Ensure Tabs look clean */
        .stTabs [data-baseweb="tab-list"] {gap: 4px;}
        
        /* Remove any forced white backgrounds on charts */
        div[data-testid="stPlotlyChart"] {
            background-color: transparent;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading weather records…")
def load_data(file) -> pd.DataFrame:
    dtypes = {
        "month": "category",
        "season": "category",
        "station_name": "category",
        "state": "category",
        "district": "category",
    }
    df = pd.read_csv(file, dtype=dtypes)

    # Robust date parsing (DD/MM/YYYY as seen in the source file)
    df["date_of_record"] = pd.to_datetime(df["date_of_record"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date_of_record"])

    numeric_cols = ["avg_temp", "min_temp", "max_temp", "wind_speed",
                     "air_pressure", "elevation", "latitude", "longitude", "rainfall"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    df["year"] = df["date_of_record"].dt.year.astype("int16")
    df["month_num"] = df["date_of_record"].dt.month.astype("int8")
    df["state_full"] = df["state"].map(STATE_NAME_MAP).fillna(df["state"].astype(str))

    return df


DEFAULT_PATH = "india_weather_rainfall_data2.csv"
try:
    raw_df = load_data(DEFAULT_PATH)
except FileNotFoundError:
    st.warning("Default CSV not found next to app.py — please upload it below.")
    uploaded = st.file_uploader("Upload india_weather_rainfall_data2.csv", type="csv")
    if uploaded is None:
        st.stop()
    raw_df = load_data(uploaded)

MIN_DATE, MAX_DATE = raw_df["date_of_record"].min().date(), raw_df["date_of_record"].max().date()


# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR — FILTERS
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.title("🌦️ Filters")

if st.sidebar.button("🔄 Reset all filters", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

date_range = st.sidebar.date_input(
    "Date range",
    value=(MIN_DATE, MAX_DATE),
    min_value=MIN_DATE,
    max_value=MAX_DATE,
    key="date_range",
)
if len(date_range) != 2:
    date_range = (MIN_DATE, MAX_DATE)

state_options = sorted(raw_df["state_full"].unique().tolist())
selected_states_full = st.sidebar.multiselect("State / UT", options=state_options, default=[], key="state_filter")
selected_states = [k for k, v in STATE_NAME_MAP.items() if v in selected_states_full] if selected_states_full else []

district_pool = raw_df[raw_df["state"].isin(selected_states)] if selected_states else raw_df
district_options = sorted(district_pool["district"].unique().tolist())
selected_districts = st.sidebar.multiselect("District", options=district_options, default=[], key="district_filter")

station_pool = district_pool[district_pool["district"].isin(selected_districts)] if selected_districts else district_pool
station_options = sorted(station_pool["station_name"].unique().tolist())
selected_stations = st.sidebar.multiselect("Station", options=station_options, default=[], key="station_filter")

selected_seasons = st.sidebar.multiselect(
    "Season", options=sorted(raw_df["season"].unique().tolist()), default=[], key="season_filter"
)
selected_months = st.sidebar.multiselect(
    "Month", options=[m for m in MONTH_ORDER if m in raw_df["month"].unique()], default=[], key="month_filter"
)

temp_min, temp_max = float(raw_df["avg_temp"].min()), float(raw_df["avg_temp"].max())
temp_range = st.sidebar.slider(
    "Avg. temperature range (°C)", min_value=float(np.floor(temp_min)), max_value=float(np.ceil(temp_max)),
    value=(float(np.floor(temp_min)), float(np.ceil(temp_max))), key="temp_filter"
)

rain_min, rain_max = float(raw_df["rainfall"].min()), float(raw_df["rainfall"].max())
rain_range = st.sidebar.slider(
    "Rainfall range (mm)", min_value=float(np.floor(rain_min)), max_value=float(np.ceil(rain_max)),
    value=(float(np.floor(rain_min)), float(np.ceil(rain_max))), key="rain_filter"
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit + Plotly · Data: 2015–2024 daily station records")


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (df["date_of_record"].dt.date >= date_range[0])
        & (df["date_of_record"].dt.date <= date_range[1])
        & (df["avg_temp"].between(temp_range[0], temp_range[1]))
        & (df["rainfall"].between(rain_range[0], rain_range[1]))
    )
    if selected_states:
        mask &= df["state"].isin(selected_states)
    if selected_districts:
        mask &= df["district"].isin(selected_districts)
    if selected_stations:
        mask &= df["station_name"].isin(selected_stations)
    if selected_seasons:
        mask &= df["season"].isin(selected_seasons)
    if selected_months:
        mask &= df["month"].isin(selected_months)
    return df.loc[mask]


df = apply_filters(raw_df)

if df.empty:
    st.error("No records match the current filter selection. Try widening your filters or hitting **Reset all filters**.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────
# HEADER + KPI ROW
# ──────────────────────────────────────────────────────────────────────────
st.title("🌧️ India Weather & Rainfall Intelligence Dashboard")
st.caption(
    f"Showing **{len(df):,}** of {len(raw_df):,} daily records · "
    f"{df['station_name'].nunique()} stations · {df['district'].nunique()} districts · "
    f"{df['state'].nunique()} states/UTs · {date_range[0]} → {date_range[1]}"
)

wettest_state = df.groupby("state_full", observed=True)["rainfall"].mean().idxmax()
hottest_station = df.groupby("station_name", observed=True)["avg_temp"].mean().idxmax()
highest_single_day = df["rainfall"].max()

# Year-over-year delta for average rainfall (vs previous full year in range, if available)
yearly_rain = df.groupby("year", observed=True)["rainfall"].mean()
rain_delta = None
if len(yearly_rain) >= 2:
    rain_delta = float(yearly_rain.iloc[-1] - yearly_rain.iloc[-2])

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Avg. Rainfall", f"{df['rainfall'].mean():.1f} mm",
          delta=f"{rain_delta:+.1f} mm YoY" if rain_delta is not None else None)
k2.metric("Wettest State", wettest_state)
k3.metric("Hottest Station", hottest_station)
k4.metric("Highest Single-Day Rainfall", f"{highest_single_day:.1f} mm")
k5.metric("Avg. Temperature", f"{df['avg_temp'].mean():.1f} °C")
k6.metric("Records", f"{len(df):,}")

st.markdown("---")


# ──────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────
def insight(text: str):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


def sample_for_scatter(data: pd.DataFrame, n: int = 6000) -> pd.DataFrame:
    """Downsample large frames for scatter/pairplot-style charts to keep the UI snappy."""
    return data.sample(n=min(n, len(data)), random_state=42) if len(data) > n else data


# ──────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────
tab_overview, tab_rain, tab_temp, tab_geo, tab_extreme = st.tabs(
    ["📊 Overview", "🌧️ Rainfall Analysis", "🌡️ Temperature & Climate", "🗺️ Geographic View", "⚠️ Extremes & Raw Data"]
)

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════
with tab_overview:
    c1, c2 = st.columns(2)

    with c1:
        # Chart 1: Yearly rainfall trend
        yearly = df.groupby("year", observed=True)["rainfall"].mean().reset_index()
        fig = px.line(yearly, x="year", y="rainfall", markers=True,
                       title="Average Rainfall Trend by Year",
                       labels={"rainfall": "Avg. Rainfall (mm)", "year": "Year"},
                       template=PLOTLY_TEMPLATE, color_discrete_sequence=[PALETTE["primary"]])
        fig.update_traces(line_width=3, marker_size=8)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Chart 2: Monthly rainfall average (chronological order)
        monthly = (df.groupby("month", observed=True)["rainfall"].mean()
                   .reindex([m for m in MONTH_ORDER if m in df["month"].unique()]).reset_index())
        fig = px.bar(monthly, x="month", y="rainfall", title="Average Rainfall by Month",
                     labels={"rainfall": "Avg. Rainfall (mm)", "month": "Month"},
                     template=PLOTLY_TEMPLATE, color="rainfall", color_continuous_scale=SEQUENTIAL_BLUES)
        st.plotly_chart(fig, use_container_width=True)

    monsoon_share = np.nan
    if "Monsoon" in df["season"].unique():
        monsoon_share = df.loc[df["season"] == "Monsoon", "rainfall"].sum() / max(df["rainfall"].sum(), 1e-9) * 100
        insight(f"**Monsoon** accounts for **{monsoon_share:.1f}%** of total rainfall in the current selection.")

    c3, c4 = st.columns(2)
    with c3:
        # Chart 3: Average rainfall by season
        season_rain = df.groupby("season", observed=True)["rainfall"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(season_rain, x="season", y="rainfall", title="Average Rainfall by Season",
                     labels={"rainfall": "Avg. Rainfall (mm)", "season": "Season"},
                     template=PLOTLY_TEMPLATE, color="season", color_discrete_sequence=CATEGORY_SEQUENCE)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        # Chart 4: Correlation heatmap
        corr_cols = ["avg_temp", "min_temp", "max_temp", "wind_speed", "air_pressure", "elevation", "rainfall"]
        corr = df[corr_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap — Weather Variables",
                         color_continuous_scale=DIVERGING, zmin=-1, zmax=1, template=PLOTLY_TEMPLATE,
                         aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

    # Plain-language correlation insight
    corr_no_self = corr.where(~np.eye(len(corr_cols), dtype=bool))
    strongest_pos = corr_no_self.stack().idxmax()
    strongest_neg = corr_no_self.stack().idxmin()
    insight(
        f"Strongest positive relationship: **{strongest_pos[0]}** ↔ **{strongest_pos[1]}** "
        f"(r = {corr_no_self.loc[strongest_pos]:.2f}). Strongest inverse relationship: "
        f"**{strongest_neg[0]}** ↔ **{strongest_neg[1]}** (r = {corr_no_self.loc[strongest_neg]:.2f})."
    )

# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — RAINFALL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
with tab_rain:
    c1, c2 = st.columns(2)
    with c1:
        # Chart 5: Rainfall distribution
        fig = px.histogram(df, x="rainfall", nbins=40, title="Distribution of Daily Rainfall",
                            labels={"rainfall": "Rainfall (mm)"}, template=PLOTLY_TEMPLATE,
                            color_discrete_sequence=[PALETTE["primary"]])
        fig.update_layout(yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Chart 6: Monthly rainfall boxplot (chronological)
        month_order_present = [m for m in MONTH_ORDER if m in df["month"].unique()]
        fig = px.box(df, x="month", y="rainfall", category_orders={"month": month_order_present},
                     title="Monthly Rainfall Distribution (Spread & Outliers)",
                     labels={"rainfall": "Rainfall (mm)", "month": "Month"},
                     template=PLOTLY_TEMPLATE, color_discrete_sequence=[PALETTE["primary"]])
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        # Chart 7: Top 10 rainiest states
        top_states = (df.groupby("state_full", observed=True)["rainfall"].mean()
                      .sort_values(ascending=False).head(10).reset_index())
        fig = px.bar(top_states, x="rainfall", y="state_full", orientation="h",
                     title="Top 10 Rainiest States (Avg. Rainfall)",
                     labels={"rainfall": "Avg. Rainfall (mm)", "state_full": "State"},
                     template=PLOTLY_TEMPLATE, color="rainfall", color_continuous_scale=SEQUENTIAL_BLUES)
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        # Chart 8: Top 15 rainiest districts
        top_districts = (df.groupby("district", observed=True)["rainfall"].mean()
                         .sort_values(ascending=False).head(15).reset_index())
        fig = px.bar(top_districts, x="rainfall", y="district", orientation="h",
                     title="Top 15 Rainiest Districts (Avg. Rainfall)",
                     labels={"rainfall": "Avg. Rainfall (mm)", "district": "District"},
                     template=PLOTLY_TEMPLATE, color="rainfall", color_continuous_scale=SEQUENTIAL_BLUES)
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    # Chart 9: State x Season rainfall heatmap
    pivot = df.pivot_table(index="state_full", columns="season", values="rainfall", aggfunc="mean")
    fig = px.imshow(pivot, text_auto=".0f", title="Average Rainfall by State & Season (mm)",
                     labels={"color": "Avg. Rainfall (mm)"}, color_continuous_scale=SEQUENTIAL_BLUES,
                     template=PLOTLY_TEMPLATE, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        # Chart 10: Wind speed vs rainfall
        s = sample_for_scatter(df)
        fig = px.scatter(s, x="wind_speed", y="rainfall", opacity=0.35, title="Wind Speed vs Rainfall",
                          labels={"wind_speed": "Wind Speed", "rainfall": "Rainfall (mm)"},
                          template=PLOTLY_TEMPLATE, color_discrete_sequence=[PALETTE["muted"]])
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        # Chart 11: Air pressure vs rainfall
        s = sample_for_scatter(df)
        fig = px.scatter(s, x="air_pressure", y="rainfall", opacity=0.35, title="Air Pressure vs Rainfall",
                          labels={"air_pressure": "Air Pressure (hPa)", "rainfall": "Rainfall (mm)"},
                          template=PLOTLY_TEMPLATE, color_discrete_sequence=[PALETTE["muted"]])
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — TEMPERATURE & CLIMATE
# ═══════════════════════════════════════════════════════════════════════
with tab_temp:
    c1, c2 = st.columns(2)
    with c1:
        # Chart 12: State-wise average temperature
        state_temp = df.groupby("state_full", observed=True)["avg_temp"].mean().sort_values().reset_index()
        fig = px.bar(state_temp, x="avg_temp", y="state_full", orientation="h",
                     title="State-wise Average Temperature",
                     labels={"avg_temp": "Avg. Temperature (°C)", "state_full": "State"},
                     template=PLOTLY_TEMPLATE, color="avg_temp", color_continuous_scale=SEQUENTIAL_ORANGES)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Chart 13: Max temperature distribution
        fig = px.histogram(df, x="max_temp", nbins=30, title="Maximum Temperature Distribution",
                            labels={"max_temp": "Max. Temperature (°C)"}, template=PLOTLY_TEMPLATE,
                            color_discrete_sequence=[PALETTE["secondary"]])
        fig.update_layout(yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    # Chart 14: Avg/Min/Max temperature trend over time (monthly granularity)
    df["year_month"] = df["date_of_record"].dt.to_period("M").dt.to_timestamp()
    trend = df.groupby("year_month", observed=True)[["avg_temp", "min_temp", "max_temp"]].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["year_month"], y=trend["max_temp"], name="Max Temp",
                              line=dict(color=PALETTE["danger"], width=2)))
    fig.add_trace(go.Scatter(x=trend["year_month"], y=trend["avg_temp"], name="Avg Temp",
                              line=dict(color=PALETTE["secondary"], width=2)))
    fig.add_trace(go.Scatter(x=trend["year_month"], y=trend["min_temp"], name="Min Temp",
                              line=dict(color=PALETTE["primary"], width=2)))
    fig.update_layout(title="Temperature Range Over Time (Monthly Avg.)", template=PLOTLY_TEMPLATE,
                       xaxis_title="Month", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        # Chart 15: Temperature vs rainfall scatter with trendline
        s = sample_for_scatter(df)
        try:
            fig = px.scatter(s, x="avg_temp", y="rainfall", trendline="ols", opacity=0.35,
                              title="Temperature vs Rainfall", template=PLOTLY_TEMPLATE,
                              labels={"avg_temp": "Avg. Temperature (°C)", "rainfall": "Rainfall (mm)"},
                              color_discrete_sequence=[PALETTE["primary"]])
        except Exception:
            # Falls back gracefully if statsmodels isn't installed
            fig = px.scatter(s, x="avg_temp", y="rainfall", opacity=0.35,
                              title="Temperature vs Rainfall", template=PLOTLY_TEMPLATE,
                              labels={"avg_temp": "Avg. Temperature (°C)", "rainfall": "Rainfall (mm)"},
                              color_discrete_sequence=[PALETTE["primary"]])
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        # Chart 16: Multivariate bubble — temp vs rainfall vs wind, sized by elevation
        s = sample_for_scatter(df, n=3000)
        fig = px.scatter(s, x="avg_temp", y="rainfall", size="wind_speed", color="season",
                          hover_data=["state_full", "station_name", "elevation"],
                          title="Temperature vs Rainfall vs Wind Speed (bubble size), by Season",
                          template=PLOTLY_TEMPLATE, color_discrete_sequence=CATEGORY_SEQUENCE,
                          labels={"avg_temp": "Avg. Temperature (°C)", "rainfall": "Rainfall (mm)"})
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 4 — GEOGRAPHIC VIEW
# ═══════════════════════════════════════════════════════════════════════
with tab_geo:
    map_metric = st.radio("Colour stations by:", ["Rainfall", "Temperature"], horizontal=True)
    station_agg = df.groupby(["station_name", "state_full", "latitude", "longitude"], observed=True).agg(
        avg_rainfall=("rainfall", "mean"), avg_temp=("avg_temp", "mean"), records=("rainfall", "count")
    ).reset_index()

    color_col = "avg_rainfall" if map_metric == "Rainfall" else "avg_temp"
    color_scale = SEQUENTIAL_BLUES if map_metric == "Rainfall" else SEQUENTIAL_ORANGES

    # Chart 17: Geographic map of stations
    fig = px.scatter_map(
        station_agg, lat="latitude", lon="longitude", color=color_col, size="records",
        hover_name="station_name", hover_data={"state_full": True, "avg_rainfall": ":.1f", "avg_temp": ":.1f"},
        color_continuous_scale=color_scale, zoom=3.6, height=560,
        title=f"Weather Stations Coloured by Avg. {map_metric}",
        template=PLOTLY_TEMPLATE,
    )

fig.update_layout(map_style="carto-positron", margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig, use_container_width=True)

insight("Marker size reflects the number of daily records for that station in the current filter selection.")

fig.update_layout(
    map_style="carto-positron",
    margin=dict(l=0, r=0, t=40, b=0)
)

st.plotly_chart(fig, use_container_width=True)

insight(
    "Marker size reflects the number of daily records for that station in the current filter selection."
)
# ═══════════════════════════════════════════════════════════════════════
# TAB 5 — EXTREMES & RAW DATA
# ═══════════════════════════════════════════════════════════════════════
with tab_extreme:
    st.subheader("Top 10 Highest Single-Day Rainfall Events")
    extreme_events = (
        df.sort_values("rainfall", ascending=False)
        .head(10)[["date_of_record", "state_full", "district", "station_name", "rainfall", "avg_temp", "wind_speed"]]
        .rename(columns={"date_of_record": "Date", "state_full": "State", "district": "District",
                          "station_name": "Station", "rainfall": "Rainfall (mm)",
                          "avg_temp": "Avg Temp (°C)", "wind_speed": "Wind Speed"})
    )
    st.dataframe(extreme_events, use_container_width=True, hide_index=True)

    st.subheader("Filtered Raw Data")
    st.dataframe(df.head(1000), use_container_width=True, hide_index=True)
    if len(df) > 1000:
        st.caption(f"Showing first 1,000 of {len(df):,} filtered rows. Use the download button for the full extract.")

    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=df.drop(columns=["year_month"], errors="ignore").to_csv(index=False).encode("utf-8"),
        file_name="filtered_weather_data.csv",
        mime="text/csv",
    )

# ──────────────────────────────────────────────────────────────────────────
# FOOTER — DATA DICTIONARY
# ──────────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this dashboard & data dictionary"):
    st.markdown(
        """
        **Source data:** Daily weather observations from 406 stations across 314 districts and 32 states/UTs
        in India, 2015–2024.

        | Column | Meaning |
        |---|---|
        | `date_of_record` | Date of observation |
        | `season` | Winter / Summer / Monsoon / etc. |
        | `avg_temp`, `min_temp`, `max_temp` | Daily temperature (°C) |
        | `wind_speed` | Daily wind speed |
        | `air_pressure` | Atmospheric pressure (hPa) |
        | `elevation` | Station elevation (m) |
        | `rainfall` | Daily rainfall (mm) |

        All charts respond to the sidebar filters. District and station lists narrow automatically
        based on the state(s) selected. Scatter/bubble charts sample up to 6,000 points for performance
        on large filter selections; aggregate charts always use the full filtered dataset.
        """
    )
