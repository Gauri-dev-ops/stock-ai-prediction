# ============================================================
# STOCKAI
# LSTM + ARIMA BASED STOCK PREDICTION SYSTEM
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping

from statsmodels.tsa.arima.model import ARIMA


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="StockAI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# STOCK LIST
# ============================================================

STOCK_NAMES = {
    "^NSEI": "NIFTY 50",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India",
    "LT.NS": "Larsen & Toubro",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC",
    "BHARTIARTL.NS": "Bharti Airtel",
    "TMPV.NS": "Tata Motors Passenger Vehicles",
    "MARUTI.NS": "Maruti Suzuki",
    "AXISBANK.NS": "Axis Bank",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "ADANIENT.NS": "Adani Enterprises"
}


# ============================================================
# SESSION STATE
# ============================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "order_review_ready" not in st.session_state:
    st.session_state.order_review_ready = False


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ StockAI Control Panel")

dark_mode = st.sidebar.toggle(
    "🌙 Dark Mode",
    value=st.session_state.dark_mode
)

st.session_state.dark_mode = dark_mode


# ============================================================
# THEME
# ============================================================

if dark_mode:

    BACKGROUND = "#0E1117"
    SIDEBAR_BACKGROUND = "#11161D"
    CARD_BACKGROUND = "#161B22"

    TEXT = "#F5F7FA"
    SECONDARY_TEXT = "#AAB4C3"

    BORDER = "#30363D"
    GRID = "#2A3340"

    PLOT_TEMPLATE = "plotly_dark"

else:

    BACKGROUND = "#F7F9FC"
    SIDEBAR_BACKGROUND = "#FFFFFF"
    CARD_BACKGROUND = "#FFFFFF"

    # Light mode: use black typography throughout for strong contrast.
    TEXT = "#000000"
    SECONDARY_TEXT = "#000000"

    BORDER = "#D9E0E7"
    GRID = "#D5D9DF"

    PLOT_TEMPLATE = "plotly_white"

# Theme-aware dashboard colors
HERO_BG_1 = CARD_BACKGROUND
HERO_BG_2 = "#EEF2FF" if not dark_mode else "#1D2430"
NAV_TEXT = TEXT
NAV_SECONDARY = SECONDARY_TEXT
NAV_BORDER = BORDER
NAV_HOVER = "#EEF2FF" if not dark_mode else "#252D38"
NAV_ACCENT = "#3B5BDB" if not dark_mode else "#A5B4FC"


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    f"""
    <style>

    /* ======================================================
       MAIN APPLICATION
       ====================================================== */

    .stApp {{
        background-color: {BACKGROUND};
        color: {TEXT};
    }}

    /* ======================================================
       STREAMLIT TOP HEADER / BLACK BAND FIX
       ====================================================== */

    header[data-testid="stHeader"] {{
        background-color: {BACKGROUND} !important;
        color: {TEXT} !important;
    }}

    header[data-testid="stHeader"] > div {{
        background-color: {BACKGROUND} !important;
    }}

    div[data-testid="stDecoration"] {{
        background-color: {BACKGROUND} !important;
        background-image: none !important;
    }}

    /* Keep the top-right Streamlit controls readable in both themes */
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] button svg {{
        color: {TEXT} !important;
        fill: {TEXT} !important;
    }}

    .main {{
        background-color: {BACKGROUND};
    }}

    .block-container {{
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }}


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {{
        background-color: {SIDEBAR_BACKGROUND};
        border-right: 1px solid {BORDER};
    }}

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {{
        color: {TEXT} !important;
    }}


    /* ======================================================
       HEADINGS
       ====================================================== */

    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT} !important;
    }}

    p {{
        color: {TEXT} !important;
    }}

    /* Streamlit captions, markdown text, labels and widget text */
    .stCaption,
    [data-testid="stCaptionContainer"],
    [data-testid="stMarkdownContainer"],
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    .stSelectbox label,
    .stDateInput label,
    .stSlider label,
    .stTextInput label {{
        color: {TEXT} !important;
    }}


    /* ======================================================
       SIDEBAR INPUTS
       ====================================================== */

    section[data-testid="stSidebar"]
    div[data-baseweb="select"] > div {{
        background-color: {CARD_BACKGROUND} !important;
        border: 1px solid {BORDER} !important;
        color: {TEXT} !important;
        border-radius: 9px !important;
    }}

    section[data-testid="stSidebar"]
    div[data-baseweb="select"] span {{
        color: {TEXT} !important;
    }}

    section[data-testid="stSidebar"] input {{
        background-color: {CARD_BACKGROUND} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
    }}


    /* ======================================================
       MAIN-PAGE INPUTS / SELECTS / BUTTONS
       Keep widgets consistent with the active StockAI theme.
       The previous version styled only sidebar widgets, which
       caused the Broker Connection controls to fall back to
       Streamlit/BaseWeb dark styling in light mode.
       ====================================================== */

    div[data-baseweb="select"] > div {{
        background-color: {CARD_BACKGROUND} !important;
        border: 1px solid {BORDER} !important;
        color: {TEXT} !important;
        border-radius: 9px !important;
    }}

    div[data-baseweb="select"] [data-baseweb="select"] {{
        background-color: {CARD_BACKGROUND} !important;
        color: {TEXT} !important;
    }}

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] svg {{
        color: {TEXT} !important;
        fill: {TEXT} !important;
    }}

    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"] {{
        background-color: {CARD_BACKGROUND} !important;
        color: {TEXT} !important;
    }}

    ul[role="listbox"] li,
    div[data-baseweb="menu"] li {{
        color: {TEXT} !important;
        background-color: {CARD_BACKGROUND} !important;
    }}

    ul[role="listbox"] li:hover,
    div[data-baseweb="menu"] li:hover {{
        background-color: {NAV_HOVER} !important;
    }}

    div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] div[data-baseweb="input"] > div,
    div[data-testid="stTextInput"] div[data-baseweb="input"] > div {{
        background-color: {CARD_BACKGROUND} !important;
        border: 1px solid {BORDER} !important;
        color: {TEXT} !important;
        border-radius: 9px !important;
    }}

    div[data-baseweb="input"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {{
        background-color: transparent !important;
        color: {TEXT} !important;
        -webkit-text-fill-color: {TEXT} !important;
    }}

    div[data-testid="stNumberInput"] button {{
        background-color: transparent !important;
        color: {TEXT} !important;
        border: 0 !important;
    }}

    div[data-testid="stNumberInput"] button:hover {{
        background-color: {NAV_HOVER} !important;
    }}

    /* Main buttons */
    div.stButton > button,
    div[data-testid="stLinkButton"] a {{
        background-color: {CARD_BACKGROUND} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 9px !important;
    }}

    div.stButton > button:hover,
    div[data-testid="stLinkButton"] a:hover {{
        background-color: {NAV_HOVER} !important;
        color: {TEXT} !important;
        border-color: {NAV_ACCENT} !important;
    }}

    div.stButton > button p,
    div[data-testid="stLinkButton"] a p,
    div.stButton > button span {{
        color: {TEXT} !important;
    }}

    /* ======================================================
       SIDEBAR BUTTONS
       ====================================================== */

    section[data-testid="stSidebar"] button {{
        border-color: {BORDER} !important;
        color: {TEXT} !important;
    }}


    /* ======================================================
       METRIC CARDS
       ====================================================== */

    div[data-testid="stMetric"] {{
        background-color: {CARD_BACKGROUND};
        border: 1px solid {BORDER};
        border-radius: 13px;
        padding: 17px;
        min-height: 110px;
    }}

    div[data-testid="stMetricLabel"] {{
        color: {SECONDARY_TEXT} !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: {TEXT} !important;
    }}


    /* ======================================================
       EXPANDERS
       ====================================================== */

    div[data-testid="stExpander"] {{
        background-color: {CARD_BACKGROUND};
        border: 1px solid {BORDER};
        border-radius: 11px;
    }}


    /* ======================================================
       TABS
       ====================================================== */

    button[data-baseweb="tab"] {{
        color: {TEXT} !important;
    }}


    /* ======================================================
       DATAFRAME
       ====================================================== */

    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 11px;
        overflow: hidden;
    }}


    /* ======================================================
       DASHBOARD / NAVIGATION
       ====================================================== */

    .hero-card {{
        background: linear-gradient(135deg, {HERO_BG_1} 0%, {HERO_BG_2} 100%);
        border: 1px solid {NAV_BORDER};
        border-radius: 18px;
        padding: 28px 30px;
        margin: 8px 0 24px 0;
        box-shadow: 0 8px 24px rgba(31, 41, 55, 0.06);
    }}

    .hero-kicker {{
        color: {NAV_SECONDARY};
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }}

    .hero-title {{
        color: {NAV_TEXT};
        font-size: 32px;
        font-weight: 750;
        margin: 0;
    }}

    .hero-subtitle {{
        color: {NAV_SECONDARY};
        font-size: 15px;
        margin-top: 8px;
    }}

    .nav-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin: 8px 0 28px 0;
    }}

    .nav-card {{
        display: block;
        text-decoration: none !important;
        background: {CARD_BACKGROUND};
        border: 1px solid #dbe3f0;
        border-radius: 14px;
        padding: 18px;
        min-height: 88px;
        box-shadow: 0 4px 14px rgba(31, 41, 55, 0.045);
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }}

    .nav-card:hover {{
        transform: translateY(-2px);
        border-color: #98aafc;
        box-shadow: 0 8px 20px rgba(31, 41, 55, 0.08);
    }}

    .nav-card-title {{
        color: {NAV_TEXT};
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 5px;
    }}

    .nav-card-text {{
        color: {NAV_SECONDARY};
        font-size: 12px;
        line-height: 1.4;
    }}

    .sidebar-nav a {{
        display: block;
        padding: 8px 10px;
        margin: 3px 0;
        border-radius: 8px;
        color: {NAV_TEXT} !important;
        text-decoration: none !important;
        font-size: 13px;
        font-weight: 550;
    }}

    .sidebar-nav a:hover {{
        background: {NAV_HOVER};
        color: {NAV_ACCENT} !important;
    }}

    @media (max-width: 900px) {{
        .nav-grid {{ grid-template-columns: 1fr 1fr; }}
    }}

    @media (max-width: 600px) {{
        .nav-grid {{ grid-template-columns: 1fr; }}
    }}


    /* ======================================================
       DIVIDER
       ====================================================== */

    hr {{
        border-color: {BORDER};
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR - STOCK
# ============================================================

stock = st.sidebar.selectbox(
    "Select Stock",
    list(STOCK_NAMES.keys()),
    format_func=lambda x: STOCK_NAMES[x]
)

company_name = STOCK_NAMES[stock]


# ============================================================
# QUICK NAVIGATION
# ============================================================

st.sidebar.markdown("### 🧭 Quick Navigation")
st.sidebar.markdown(
    """
    <div class="sidebar-nav">
        <a href="#stockai-dashboard">🏠 Dashboard</a>
        <a href="#market-overview">📊 Market Overview</a>
        <a href="#price-history">🕯️ Price History</a>
        <a href="#technical-analysis">📐 Technical Analysis</a>
        <a href="#ai-prediction">🤖 AI Prediction</a>
        <a href="#stockai-signal">🧠 StockAI Signal</a>
        <a href="#ai-stock-forecast">📈 AI Forecast</a>
        <a href="#future-predictions">📋 Future Predictions</a>
        <a href="#broker-connection">🔗 Broker Connection</a>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR - HISTORICAL START
# ============================================================

historical_start = st.sidebar.date_input(
    "Historical Data Start",
    value=pd.to_datetime("2022-04-01")
)


# ============================================================
# SIDEBAR - FORECAST LENGTH
# ============================================================

prediction_days = st.sidebar.select_slider(
    "Prediction Days",
    options=[15, 30, 60, 90, 120],
    value=30
)


# ============================================================
# SIDEBAR - UPDATE
# ============================================================

update_forecast = st.sidebar.button(
    "🔄 Update Forecast",
    use_container_width=True
)

if update_forecast:

    st.cache_data.clear()
    st.rerun()


# ============================================================
# SIDEBAR INFORMATION
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "StockAI combines historical market data, "
    "technical indicators, LSTM forecasting and "
    "ARIMA forecasting."
)


# ============================================================
# MAIN HEADER
# ============================================================

st.title("📈 StockAI")

st.subheader(
    f"{company_name} — AI Market Analysis"
)

st.markdown(
    "### LSTM + ARIMA Based Stock Prediction System"
)

st.caption(
    "AI-powered historical analysis, technical indicators "
    "and future price forecasting."
)


# ============================================================
# DATA DOWNLOAD
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def download_stock_data(
    stock_symbol,
    start_date
):

    start_date = pd.Timestamp(
        start_date
    )

    end_date = (
        pd.Timestamp.today().normalize()
        + pd.Timedelta(days=1)
    )

    stock_data = yf.download(
        stock_symbol,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    return stock_data


# ============================================================
# DOWNLOAD DATA
# ============================================================

with st.spinner(
    "📡 Downloading latest market data..."
):

    data = download_stock_data(
        stock,
        historical_start
    )


# ============================================================
# DATA VALIDATION
# ============================================================

if data is None or data.empty:

    st.error(
        "❌ No market data was downloaded. "
        "Please check your internet connection "
        "or select another stock."
    )

    st.stop()


# ============================================================
# CLEAN YFINANCE DATA
# ============================================================

data = data.reset_index()


if isinstance(
    data.columns,
    pd.MultiIndex
):

    data.columns = (
        data.columns
        .get_level_values(0)
    )


data = data.loc[
    :,
    ~data.columns.duplicated()
]


if "Date" not in data.columns:

    data.rename(
        columns={
            data.columns[0]: "Date"
        },
        inplace=True
    )


required_columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]


available_columns = [
    column
    for column in required_columns
    if column in data.columns
]


data = data[
    available_columns
]


numeric_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]


for column in numeric_columns:

    if column in data.columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )


data.dropna(
    subset=["Close"],
    inplace=True
)


data["Date"] = pd.to_datetime(
    data["Date"]
)


data.sort_values(
    "Date",
    inplace=True
)


data.reset_index(
    drop=True,
    inplace=True
)


# ============================================================
# MINIMUM DATA CHECK
# ============================================================

if len(data) < 150:

    st.error(
        "❌ Not enough historical data is available "
        "for reliable LSTM and ARIMA modelling. "
        "Please select an earlier start date."
    )

    st.stop()


# ============================================================
# DATA STATUS
# ============================================================

actual_start = (
    data["Date"]
    .min()
    .date()
)

actual_end = (
    data["Date"]
    .max()
    .date()
)


st.success(
    f"✅ Loaded {len(data):,} trading records "
    f"from {actual_start} to {actual_end}"
)


# ============================================================
# MARKET OVERVIEW
# ============================================================

latest_price = float(
    data["Close"].iloc[-1]
)

previous_price = float(
    data["Close"].iloc[-2]
)

daily_change = (
    latest_price -
    previous_price
)

daily_change_percent = (
    daily_change /
    previous_price
) * 100


highest_price = float(
    data["High"].max()
)

lowest_price = float(
    data["Low"].min()
)


# ============================================================
# STOCKAI DASHBOARD
# ============================================================

st.markdown('<div id="stockai-dashboard"></div>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-kicker">StockAI • Final Project Dashboard</div>
        <div class="hero-title">{company_name} — AI Market Analysis</div>
        <div class="hero-subtitle">LSTM + ARIMA based stock prediction, historical analysis and technical indicators.</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="nav-grid">
        <a class="nav-card" href="#market-overview">
            <div class="nav-card-title">📊 Market Overview</div>
            <div class="nav-card-text">Current price, daily movement and historical range.</div>
        </a>
        <a class="nav-card" href="#price-history">
            <div class="nav-card-title">🕯️ Price History</div>
            <div class="nav-card-text">Interactive candlestick view of market history.</div>
        </a>
        <a class="nav-card" href="#technical-analysis">
            <div class="nav-card-title">📐 Technical Analysis</div>
            <div class="nav-card-text">Moving averages, RSI and MACD indicators.</div>
        </a>
        <a class="nav-card" href="#ai-prediction">
            <div class="nav-card-title">🤖 AI Prediction</div>
            <div class="nav-card-text">LSTM validation and model performance metrics.</div>
        </a>
        <a class="nav-card" href="#ai-stock-forecast">
            <div class="nav-card-title">📈 AI Forecast</div>
            <div class="nav-card-text">Future LSTM and ARIMA forecast comparison.</div>
        </a>
        <a class="nav-card" href="#future-predictions">
            <div class="nav-card-title">📋 Future Predictions</div>
            <div class="nav-card-text">Day-by-day forecast values and expected movement.</div>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"**Data coverage:** {len(data):,} trading records  •  **From:** {data['Date'].min().strftime('%d %b %Y')}  •  **To:** {data['Date'].max().strftime('%d %b %Y')}"
)

st.markdown('<div id="market-overview"></div>', unsafe_allow_html=True)

st.subheader(
    "📊 Market Overview"
)


market_col1, market_col2, market_col3, market_col4 = (
    st.columns(4)
)


market_col1.metric(
    "Current Price",
    f"₹{latest_price:,.2f}"
)


market_col2.metric(
    "Daily Change",
    f"₹{daily_change:,.2f}",
    f"{daily_change_percent:+.2f}%"
)


market_col3.metric(
    "Highest Price",
    f"₹{highest_price:,.2f}"
)


market_col4.metric(
    "Lowest Price",
    f"₹{lowest_price:,.2f}"
)


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

data["MA50"] = (
    data["Close"]
    .rolling(50)
    .mean()
)


data["MA100"] = (
    data["Close"]
    .rolling(100)
    .mean()
)


data["Daily Return"] = (
    data["Close"]
    .pct_change()
)


# ============================================================
# RSI
# ============================================================

delta = (
    data["Close"]
    .diff()
)


gain = (
    delta
    .clip(lower=0)
    .rolling(14)
    .mean()
)


loss = (
    -delta
    .clip(upper=0)
    .rolling(14)
    .mean()
)


rs = gain / loss.replace(
    0,
    np.nan
)


data["RSI"] = (
    100 -
    (
        100 /
        (1 + rs)
    )
)


# ============================================================
# MACD
# ============================================================

ema12 = (
    data["Close"]
    .ewm(
        span=12,
        adjust=False
    )
    .mean()
)


ema26 = (
    data["Close"]
    .ewm(
        span=26,
        adjust=False
    )
    .mean()
)


data["MACD"] = (
    ema12 -
    ema26
)


data["Signal_Line"] = (
    data["MACD"]
    .ewm(
        span=9,
        adjust=False
    )
    .mean()
)


# ============================================================
# HISTORICAL DATA
# ============================================================

with st.expander(
    "📋 View Historical Stock Data"
):

    historical_display = data.copy()

    st.dataframe(
        historical_display,
        use_container_width=True,
        height=420
    )


# ============================================================
# CHART HELPER
# ============================================================

def style_chart(
    figure,
    title,
    height=550
):
    """Apply StockAI visual styling without changing chart data or model logic."""

    plot_background = (
        "#0E1117" if dark_mode else "#FFFFFF"
    )

    axis_color = (
        "#F5F7FA" if dark_mode else "#000000"
    )

    tick_color = (
        "#D7DCE5" if dark_mode else "#000000"
    )

    grid_color = (
        "#303641" if dark_mode else "#D5D9DF"
    )

    border_color = (
        "#59606C" if dark_mode else "#000000"
    )

    figure.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        title=dict(
            text=title,
            font=dict(
                family="Arial, sans-serif",
                size=21,
                color=axis_color
            )
        ),
        font=dict(
            family="Arial, sans-serif",
            color=axis_color,
            size=13
        ),
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=plot_background,
        xaxis=dict(
            title="Date",
            title_font=dict(
                family="Arial, sans-serif",
                size=13,
                color=axis_color
            ),
            tickfont=dict(
                family="Arial, sans-serif",
                size=12,
                color=tick_color
            ),
            showgrid=True,
            gridcolor=grid_color,
            zeroline=False,
            showline=True,
            linecolor=border_color,
            mirror=False
        ),
        yaxis=dict(
            title="Price",
            title_font=dict(
                family="Arial, sans-serif",
                size=13,
                color=axis_color
            ),
            tickfont=dict(
                family="Arial, sans-serif",
                size=12,
                color=tick_color
            ),
            showgrid=True,
            gridcolor=grid_color,
            zeroline=False,
            showline=True,
            linecolor=border_color,
            mirror=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color=axis_color
            ),
            bgcolor=(
                "rgba(14,17,23,0.80)"
                if dark_mode
                else "rgba(255,255,255,0.92)"
            ),
            bordercolor=(
                "#303641" if dark_mode else "#D5D9DF"
            ),
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor=(
                "#161B22" if dark_mode else "#FFFFFF"
            ),
            bordercolor=border_color,
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color=axis_color
            )
        ),
        hovermode="x unified",
        margin=dict(
            l=60,
            r=30,
            t=90,
            b=60
        )
    )

    return figure


def apply_plotly_theme(figure):
    """Apply theme colors to an already-configured Plotly figure."""

    axis_color = "#F5F7FA" if dark_mode else "#000000"
    tick_color = "#D7DCE5" if dark_mode else "#000000"
    grid_color = "#303641" if dark_mode else "#D5D9DF"
    border_color = "#59606C" if dark_mode else "#000000"
    plot_background = "#0E1117" if dark_mode else "#FFFFFF"

    figure.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=plot_background,
        font=dict(
            family="Arial, sans-serif",
            color=axis_color,
            size=13
        ),
        title_font=dict(
            family="Arial, sans-serif",
            color=axis_color
        ),
        xaxis=dict(
            title_font=dict(
                family="Arial, sans-serif",
                color=axis_color,
                size=13
            ),
            tickfont=dict(
                family="Arial, sans-serif",
                color=tick_color,
                size=12
            ),
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            linecolor=border_color,
            showline=True
        ),
        yaxis=dict(
            title_font=dict(
                family="Arial, sans-serif",
                color=axis_color,
                size=13
            ),
            tickfont=dict(
                family="Arial, sans-serif",
                color=tick_color,
                size=12
            ),
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            linecolor=border_color,
            showline=True
        ),
        legend=dict(
            font=dict(
                family="Arial, sans-serif",
                color=axis_color,
                size=12
            ),
            bgcolor=(
                "rgba(14,17,23,0.80)"
                if dark_mode
                else "rgba(255,255,255,0.92)"
            ),
            bordercolor=(
                "#303641" if dark_mode else "#D5D9DF"
            ),
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor=(
                "#161B22" if dark_mode else "#FFFFFF"
            ),
            bordercolor=border_color,
            font=dict(
                family="Arial, sans-serif",
                color=axis_color,
                size=12
            )
        )
    )

    # Keep annotation text readable in both themes.
    for annotation in figure.layout.annotations:
        annotation.font = dict(
            family="Arial, sans-serif",
            color=axis_color,
            size=(
                annotation.font.size
                if annotation.font and annotation.font.size
                else 12
            )
        )

    return figure


# ============================================================
# PRICE HISTORY
# ============================================================

st.markdown('<div id="price-history"></div>', unsafe_allow_html=True)

st.subheader(
    "🕯️ Price History"
)


price_chart = go.Figure()


price_chart.add_trace(
    go.Candlestick(
        x=data["Date"],
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Price"
    )
)


price_chart.update_layout(
    template=PLOT_TEMPLATE,
    height=580,
    title=f"{company_name} Price History",
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    paper_bgcolor=BACKGROUND,
    plot_bgcolor=("#0E1117" if dark_mode else "#FFFFFF"),
    font=dict(color=TEXT),
    hovermode="x unified"
)

price_chart = apply_plotly_theme(price_chart)

st.plotly_chart(
    price_chart,
    use_container_width=True,
    config={
        "displaylogo": False,
        "responsive": True
    }
)


# ============================================================
# TECHNICAL ANALYSIS
# ============================================================

st.markdown('<div id="technical-analysis"></div>', unsafe_allow_html=True)

st.subheader(
    "📐 Technical Analysis"
)


tab_ma, tab_rsi, tab_macd = st.tabs(
    [
        "Moving Averages",
        "RSI",
        "MACD"
    ]
)


# ============================================================
# MOVING AVERAGES
# ============================================================

with tab_ma:

    ma_chart = go.Figure()


    ma_chart.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Close"],
            mode="lines",
            name="Close Price",
            line=dict(
                width=2
            )
        )
    )


    ma_chart.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["MA50"],
            mode="lines",
            name="50 Day MA",
            line=dict(
                width=2
            )
        )
    )


    ma_chart.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["MA100"],
            mode="lines",
            name="100 Day MA",
            line=dict(
                width=2
            )
        )
    )


    ma_chart = style_chart(
        ma_chart,
        "Moving Average Analysis",
        520
    )


    st.plotly_chart(
        ma_chart,
        use_container_width=True,
        config={
            "displaylogo": False
        }
    )


# ============================================================
# RSI
# ============================================================

with tab_rsi:

    rsi_chart = go.Figure()


    rsi_chart.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["RSI"],
            mode="lines",
            name="RSI",
            line=dict(
                width=2
            )
        )
    )


    rsi_chart.add_hline(
        y=70,
        line_dash="dash",
        annotation_text="Overbought"
    )


    rsi_chart.add_hline(
        y=30,
        line_dash="dash",
        annotation_text="Oversold"
    )


    rsi_chart.update_layout(
        template=PLOT_TEMPLATE,
        height=450,
        title="Relative Strength Index",
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis=dict(
            range=[0, 100],
            gridcolor=GRID
        ),
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=("#0E1117" if dark_mode else "#FFFFFF"),
        font=dict(color=TEXT),
        hovermode="x unified"
    )

    rsi_chart = apply_plotly_theme(rsi_chart)

    st.plotly_chart(
        rsi_chart,
        use_container_width=True,
        config={
            "displaylogo": False
        }
    )


# ============================================================
# MACD
# ============================================================

with tab_macd:

    macd_chart = go.Figure()


    macd_chart.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["MACD"],
            mode="lines",
            name="MACD",
            line=dict(
                width=2
            )
        )
    )


    macd_chart.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Signal_Line"],
            mode="lines",
            name="Signal Line",
            line=dict(
                width=2,
                dash="dash"
            )
        )
    )


    macd_chart.update_layout(
        template=PLOT_TEMPLATE,
        height=450,
        title="MACD Analysis",
        xaxis_title="Date",
        yaxis_title="MACD",
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=("#0E1117" if dark_mode else "#FFFFFF"),
        font=dict(color=TEXT),
        hovermode="x unified"
    )

    macd_chart = apply_plotly_theme(macd_chart)

    st.plotly_chart(
        macd_chart,
        use_container_width=True,
        config={
            "displaylogo": False
        }
    )


# ============================================================
# MODEL FUNCTION
# ============================================================

@st.cache_data(
    show_spinner=False,
    ttl=1800
)
def generate_predictions(
    close_values,
    prediction_days
):

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    np.random.seed(42)
    tf.random.set_seed(42)


    # --------------------------------------------------------
    # PREPARE PRICE DATA
    # --------------------------------------------------------

    prices = np.asarray(
        close_values,
        dtype=np.float64
    ).reshape(-1, 1)


    lookback = 60


    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    training_length = int(
        len(prices) * 0.80
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # FIT SCALER ONLY ON TRAINING DATA.
    #
    # This prevents validation/future information
    # from influencing the transformation.
    # --------------------------------------------------------

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )


    scaler.fit(
        prices[
            :training_length
        ]
    )


    scaled_data = scaler.transform(
        prices
    )


    train_scaled = scaled_data[
        :training_length
    ]


    # --------------------------------------------------------
    # CREATE LSTM TRAINING SEQUENCES
    # --------------------------------------------------------

    x_train = []
    y_train = []


    for i in range(
        lookback,
        len(train_scaled)
    ):

        x_train.append(
            train_scaled[
                i - lookback:i,
                0
            ]
        )

        y_train.append(
            train_scaled[
                i,
                0
            ]
        )


    x_train = np.array(
        x_train,
        dtype=np.float32
    )


    y_train = np.array(
        y_train,
        dtype=np.float32
    )


    x_train = x_train.reshape(
        x_train.shape[0],
        x_train.shape[1],
        1
    )


    # --------------------------------------------------------
    # BUILD LSTM
    # --------------------------------------------------------

    model = Sequential(
        [
            Input(
                shape=(
                    lookback,
                    1
                )
            ),

            LSTM(
                64,
                return_sequences=True
            ),

            LSTM(
                64
            ),

            Dense(
                32,
                activation="relu"
            ),

            Dense(
                1
            )
        ]
    )


    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    early_stopping = EarlyStopping(
        monitor="loss",
        patience=3,
        restore_best_weights=True
    )


    # --------------------------------------------------------
    # TRAIN LSTM
    # --------------------------------------------------------

    model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=20,
        verbose=0,
        callbacks=[
            early_stopping
        ]
    )


    # --------------------------------------------------------
    # VALIDATION DATA
    # --------------------------------------------------------

    test_scaled = scaled_data[
        training_length - lookback:
    ]


    x_test = []


    for i in range(
        lookback,
        len(test_scaled)
    ):

        x_test.append(
            test_scaled[
                i - lookback:i,
                0
            ]
        )


    x_test = np.array(
        x_test,
        dtype=np.float32
    )


    x_test = x_test.reshape(
        x_test.shape[0],
        x_test.shape[1],
        1
    )


    # --------------------------------------------------------
    # LSTM VALIDATION PREDICTIONS
    # --------------------------------------------------------

    validation_predictions_scaled = (
        model.predict(
            x_test,
            verbose=0
        )
    )


    validation_predictions = (
        scaler.inverse_transform(
            validation_predictions_scaled
        )
        .flatten()
    )


    # --------------------------------------------------------
    # FUTURE LSTM FORECAST
    # --------------------------------------------------------

    future_input = (
        scaled_data[-lookback:]
        .reshape(
            1,
            lookback,
            1
        )
    )


    future_scaled = []


    for _ in range(
        prediction_days
    ):

        next_prediction = (
            model.predict(
                future_input,
                verbose=0
            )[0, 0]
        )


        future_scaled.append(
            next_prediction
        )


        next_value = np.array(
            next_prediction,
            dtype=np.float32
        ).reshape(
            1,
            1,
            1
        )


        future_input = np.concatenate(
            [
                future_input[:, 1:, :],
                next_value
            ],
            axis=1
        )


    future_lstm = (
        scaler.inverse_transform(
            np.array(
                future_scaled
            ).reshape(-1, 1)
        )
        .flatten()
    )


    # --------------------------------------------------------
    # ARIMA HISTORICAL VALIDATION
    # --------------------------------------------------------
    # Fit ARIMA only on the training portion first, then forecast
    # the same holdout period used for the LSTM validation.
    # This gives StockAI a comparable historical accuracy measure.

    train_prices = prices[:training_length].flatten()
    validation_actual = prices[training_length:].flatten()

    try:
        arima_validation_model = ARIMA(
            train_prices,
            order=(5, 1, 2)
        )

        arima_validation_fit = arima_validation_model.fit()

        arima_validation = arima_validation_fit.forecast(
            steps=len(validation_actual)
        )

        arima_validation = np.asarray(
            arima_validation,
            dtype=float
        )

    except Exception:
        # Keep the application running if a particular series does not
        # converge cleanly during historical ARIMA validation.
        arima_validation = np.full(
            len(validation_actual),
            np.nan,
            dtype=float
        )


    # --------------------------------------------------------
    # ARIMA FUTURE FORECAST
    # --------------------------------------------------------

    arima_model = ARIMA(
        prices.flatten(),
        order=(5, 1, 2)
    )


    arima_fit = arima_model.fit()


    arima_forecast = (
        arima_fit
        .forecast(
            steps=prediction_days
        )
    )


    arima_forecast = np.asarray(
        arima_forecast,
        dtype=float
    )


    return (
        validation_predictions,
        future_lstm,
        arima_forecast,
        arima_validation
    )


# ============================================================
# MODEL DATA CHECK
# ============================================================

if len(data) < 150:

    st.error(
        "Not enough data for modelling."
    )

    st.stop()


# ============================================================
# AI MODEL SECTION
# ============================================================

st.markdown('<div id="ai-prediction"></div>', unsafe_allow_html=True)

st.subheader(
    "🤖 AI Prediction"
)


with st.spinner(
    "🧠 Training LSTM and ARIMA models..."
):

    (
        lstm_validation,
        future_lstm,
        arima_forecast,
        arima_validation
    ) = generate_predictions(
        tuple(
            data["Close"].astype(float).values
        ),
        prediction_days
    )


# ============================================================
# VALIDATION DATA
# ============================================================

training_length = int(
    len(data) * 0.80
)


train_data_display = (
    data.iloc[
        :training_length
    ]
    .copy()
)


validation_data = (
    data.iloc[
        training_length:
    ]
    .copy()
)


validation_data[
    "LSTM Prediction"
] = lstm_validation

validation_data[
    "ARIMA Prediction"
] = arima_validation


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def calculate_mape(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = (
        np.isfinite(actual)
        & np.isfinite(predicted)
        & (np.abs(actual) > 1e-9)
    )

    if not np.any(mask):
        return np.nan

    return float(
        np.mean(
            np.abs(
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        ) * 100
    )


actual_validation = validation_data["Close"].astype(float).values
lstm_validation_values = validation_data["LSTM Prediction"].astype(float).values
arima_validation_values = validation_data["ARIMA Prediction"].astype(float).values


lstm_rmse = np.sqrt(
    mean_squared_error(
        actual_validation,
        lstm_validation_values
    )
)

lstm_mae = mean_absolute_error(
    actual_validation,
    lstm_validation_values
)

lstm_mape = calculate_mape(
    actual_validation,
    lstm_validation_values
)

arima_valid_mask = (
    np.isfinite(actual_validation)
    & np.isfinite(arima_validation_values)
)

if np.any(arima_valid_mask):

    arima_rmse = np.sqrt(
        mean_squared_error(
            actual_validation[arima_valid_mask],
            arima_validation_values[arima_valid_mask]
        )
    )

    arima_mae = mean_absolute_error(
        actual_validation[arima_valid_mask],
        arima_validation_values[arima_valid_mask]
    )

    arima_mape = calculate_mape(
        actual_validation[arima_valid_mask],
        arima_validation_values[arima_valid_mask]
    )

else:

    arima_rmse = np.nan
    arima_mae = np.nan
    arima_mape = np.nan


st.subheader(
    "📊 Model Performance"
)


performance_col1, performance_col2, performance_col3, performance_col4 = (
    st.columns(4)
)


performance_col1.metric(
    "LSTM RMSE",
    f"{lstm_rmse:,.2f}"
)


performance_col2.metric(
    "LSTM MAE",
    f"{lstm_mae:,.2f}"
)


performance_col3.metric(
    "ARIMA RMSE",
    f"{arima_rmse:,.2f}" if np.isfinite(arima_rmse) else "N/A"
)


performance_col4.metric(
    "ARIMA MAE",
    f"{arima_mae:,.2f}" if np.isfinite(arima_mae) else "N/A"
)


performance_info_col1, performance_info_col2 = st.columns(2)

performance_info_col1.caption(
    f"LSTM historical validation MAPE: {lstm_mape:.2f}%"
    if np.isfinite(lstm_mape) else
    "LSTM historical validation MAPE: N/A"
)

performance_info_col2.caption(
    f"ARIMA historical validation MAPE: {arima_mape:.2f}%"
    if np.isfinite(arima_mape) else
    "ARIMA historical validation MAPE: N/A"
)


# ============================================================
# LSTM VALIDATION CHART
# ============================================================

validation_chart = go.Figure()


validation_chart.add_trace(
    go.Scatter(
        x=train_data_display["Date"],
        y=train_data_display["Close"],
        mode="lines",
        name="Training Data",
        line=dict(
            width=2
        )
    )
)


validation_chart.add_trace(
    go.Scatter(
        x=validation_data["Date"],
        y=validation_data["Close"],
        mode="lines",
        name="Actual Price",
        line=dict(
            width=2.5
        )
    )
)


validation_chart.add_trace(
    go.Scatter(
        x=validation_data["Date"],
        y=validation_data[
            "LSTM Prediction"
        ],
        mode="lines",
        name="LSTM Prediction",
        line=dict(
            width=2,
            dash="dash"
        )
    )
)


validation_chart.update_layout(
    template=PLOT_TEMPLATE,
    height=560,
    title="LSTM Historical Validation",
    xaxis_title="Date",
    yaxis_title="Price",
    paper_bgcolor=BACKGROUND,
    plot_bgcolor=("#0E1117" if dark_mode else "#FFFFFF"),
    font=dict(color=TEXT),
    hovermode="x unified"
)

validation_chart = apply_plotly_theme(validation_chart)

st.plotly_chart(
    validation_chart,
    use_container_width=True,
    config={
        "displaylogo": False,
        "responsive": True
    }
)


# ============================================================
# STOCKAI SIGNAL ENGINE
# ============================================================

st.markdown('<div id="stockai-signal"></div>', unsafe_allow_html=True)

st.subheader(
    "🧠 StockAI Decision Engine"
)

st.caption(
    "A transparent decision-support layer combining model forecasts, "
    "historical validation and technical indicators. It is not a trade instruction."
)


# ------------------------------------------------------------
# FORECAST CONSENSUS
# ------------------------------------------------------------

lstm_final_signal = float(future_lstm[-1])
arima_final_signal = float(arima_forecast[-1])

lstm_signal_change = (
    (lstm_final_signal - latest_price) / latest_price * 100
)

arima_signal_change = (
    (arima_final_signal - latest_price) / latest_price * 100
)

combined_forecast = float(
    np.mean([lstm_final_signal, arima_final_signal])
)

combined_change = (
    (combined_forecast - latest_price) / latest_price * 100
)

model_spread = abs(
    lstm_signal_change - arima_signal_change
)


# ------------------------------------------------------------
# TECHNICAL SIGNALS
# ------------------------------------------------------------

current_rsi = float(data["RSI"].iloc[-1]) if np.isfinite(data["RSI"].iloc[-1]) else np.nan
current_macd = float(data["MACD"].iloc[-1]) if np.isfinite(data["MACD"].iloc[-1]) else np.nan
current_signal_line = float(data["Signal_Line"].iloc[-1]) if np.isfinite(data["Signal_Line"].iloc[-1]) else np.nan
current_ma50 = float(data["MA50"].iloc[-1]) if np.isfinite(data["MA50"].iloc[-1]) else np.nan
current_ma100 = float(data["MA100"].iloc[-1]) if np.isfinite(data["MA100"].iloc[-1]) else np.nan

recent_returns = data["Daily Return"].dropna().astype(float)
annualised_volatility = (
    recent_returns.tail(60).std() * np.sqrt(252) * 100
    if len(recent_returns) >= 10 else np.nan
)


# ------------------------------------------------------------
# TRANSPARENT SCORE
# ------------------------------------------------------------
# The score is deliberately based on observable components rather than
# a hidden probability. It should be treated as decision support only.

signal_score = 50.0
signal_reasons = []
risk_factors = []


# Forecast direction: maximum +/- 25 points.
if combined_change > 5:
    signal_score += 25
    signal_reasons.append(
        f"Combined model forecast is {combined_change:+.2f}% above the current price."
    )
elif combined_change > 2:
    signal_score += 15
    signal_reasons.append(
        f"Combined model forecast is moderately positive at {combined_change:+.2f}%."
    )
elif combined_change > 0:
    signal_score += 8
    signal_reasons.append(
        f"Combined model forecast is slightly positive at {combined_change:+.2f}%."
    )
elif combined_change < -5:
    signal_score -= 25
    signal_reasons.append(
        f"Combined model forecast is {combined_change:+.2f}% below the current price."
    )
elif combined_change < -2:
    signal_score -= 15
    signal_reasons.append(
        f"Combined model forecast is moderately negative at {combined_change:+.2f}%."
    )
else:
    signal_score -= 5
    signal_reasons.append(
        f"Combined model forecast is close to the current price at {combined_change:+.2f}%."
    )


# Model agreement: lower spread = more consistent models.
if model_spread <= 2:
    signal_score += 15
    signal_reasons.append(
        f"LSTM and ARIMA are closely aligned (forecast spread {model_spread:.2f} percentage points)."
    )
elif model_spread <= 5:
    signal_score += 5
    signal_reasons.append(
        f"LSTM and ARIMA show moderate agreement (spread {model_spread:.2f} percentage points)."
    )
else:
    signal_score -= 12
    risk_factors.append(
        f"The two models disagree materially (spread {model_spread:.2f} percentage points)."
    )


# LSTM historical accuracy.
if np.isfinite(lstm_mape):
    if lstm_mape <= 5:
        signal_score += 10
    elif lstm_mape <= 10:
        signal_score += 5
    elif lstm_mape > 15:
        signal_score -= 8
        risk_factors.append(
            f"LSTM historical validation MAPE is relatively high at {lstm_mape:.2f}%."
        )


# ARIMA historical accuracy.
if np.isfinite(arima_mape):
    if arima_mape <= 5:
        signal_score += 10
    elif arima_mape <= 10:
        signal_score += 5
    elif arima_mape > 15:
        signal_score -= 8
        risk_factors.append(
            f"ARIMA historical validation MAPE is relatively high at {arima_mape:.2f}%."
        )
else:
    risk_factors.append("ARIMA historical validation was unavailable for this series.")


# Moving-average structure.
if np.isfinite(current_ma50):
    if latest_price > current_ma50:
        signal_score += 7
        signal_reasons.append("Price is above the 50-day moving average.")
    else:
        signal_score -= 7
        risk_factors.append("Price is below the 50-day moving average.")

if np.isfinite(current_ma100):
    if latest_price > current_ma100:
        signal_score += 5
        signal_reasons.append("Price is above the 100-day moving average.")
    else:
        signal_score -= 5
        risk_factors.append("Price is below the 100-day moving average.")


# RSI context: avoid treating an extreme RSI as automatically bullish.
if np.isfinite(current_rsi):
    if 50 <= current_rsi < 70:
        signal_score += 5
        signal_reasons.append(f"RSI is {current_rsi:.1f}, indicating positive momentum without an extreme reading.")
    elif current_rsi >= 70:
        signal_score -= 5
        risk_factors.append(f"RSI is {current_rsi:.1f}, which is in an overbought range.")
    elif current_rsi < 30:
        signal_score -= 2
        risk_factors.append(f"RSI is {current_rsi:.1f}, indicating a weak/oversold momentum condition.")
    else:
        signal_reasons.append(f"RSI is {current_rsi:.1f}, giving a neutral momentum reading.")


# MACD context.
if np.isfinite(current_macd) and np.isfinite(current_signal_line):
    if current_macd > current_signal_line:
        signal_score += 5
        signal_reasons.append("MACD is above its signal line.")
    else:
        signal_score -= 5
        risk_factors.append("MACD is below its signal line.")


# Volatility penalty.
if np.isfinite(annualised_volatility):
    if annualised_volatility > 45:
        signal_score -= 10
        risk_factors.append(
            f"Recent annualised volatility is elevated at approximately {annualised_volatility:.1f}%."
        )
    elif annualised_volatility > 30:
        signal_score -= 5
        risk_factors.append(
            f"Recent annualised volatility is relatively high at approximately {annualised_volatility:.1f}%."
        )

signal_score = float(np.clip(signal_score, 0, 100))


# Classification deliberately avoids a direct BUY/SELL instruction.
if signal_score >= 65 and combined_change > 0:
    signal_label = "POSITIVE OUTLOOK"
elif signal_score <= 40 and combined_change < 0:
    signal_label = "CAUTION OUTLOOK"
else:
    signal_label = "NEUTRAL OUTLOOK"


# ------------------------------------------------------------
# SIGNAL DASHBOARD
# ------------------------------------------------------------

signal_col1, signal_col2, signal_col3, signal_col4 = st.columns(4)

signal_col1.metric(
    "Decision Score",
    f"{signal_score:.0f}/100"
)

signal_col2.metric(
    "Combined Forecast",
    f"₹{combined_forecast:,.2f}",
    f"{combined_change:+.2f}%"
)

signal_col3.metric(
    "Model Agreement",
    f"{model_spread:.2f}%",
    "Lower = closer"
)

signal_col4.metric(
    "Annualised Volatility",
    f"{annualised_volatility:.1f}%" if np.isfinite(annualised_volatility) else "N/A"
)


st.markdown(
    f"""
    <div style="
        margin: 18px 0;
        padding: 22px;
        border: 1px solid {BORDER};
        border-radius: 14px;
        background: {CARD_BACKGROUND};
        text-align: center;
    ">
        <div style="font-size: 13px; color: {SECONDARY_TEXT}; margin-bottom: 6px;">
            STOCKAI SIGNAL
        </div>
        <div style="font-size: 28px; font-weight: 800; color: {TEXT};">
            {signal_label}
        </div>
        <div style="font-size: 13px; color: {SECONDARY_TEXT}; margin-top: 8px;">
            Based on forecast direction, model agreement, historical validation and technical indicators.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


reason_col, risk_col = st.columns(2)

with reason_col:
    st.markdown("### ✅ Positive / Supporting Factors")
    if signal_reasons:
        for reason in signal_reasons:
            st.markdown(f"- {reason}")
    else:
        st.write("No strong supporting factor identified.")

with risk_col:
    st.markdown("### ⚠️ Risk Factors")
    if risk_factors:
        for risk in risk_factors:
            st.markdown(f"- {risk}")
    else:
        st.write("No major risk flag identified by the current rules.")


# ============================================================
# BROKER CONNECTION / ORDER WORKFLOW
# ============================================================

st.markdown('<div id="broker-connection"></div>', unsafe_allow_html=True)
st.subheader("🔗 Broker workspace")
st.caption(
    "Review a potential order and open your broker’s official developer portal. "
    "This workspace does not connect to a trading account or place live orders."
)

st.markdown(
    f"""
    <style>
      .broker-eyebrow {{ color: {NAV_ACCENT}; font-size: 11px; font-weight: 800; letter-spacing: .09em; text-transform: uppercase; margin-bottom: 7px; }}
      .broker-title {{ color: {TEXT}; font-size: 21px; font-weight: 750; margin-bottom: 5px; }}
      .broker-muted {{ color: {SECONDARY_TEXT}; font-size: 13px; line-height: 1.55; }}
      .broker-pill {{ display: inline-block; padding: 5px 9px; border-radius: 999px; border: 1px solid {BORDER}; color: {TEXT}; font-size: 11px; font-weight: 650; margin: 8px 0 14px 0; }}
      .broker-summary {{ background: {HERO_BG_2}; border: 1px solid {BORDER}; border-radius: 12px; padding: 15px 16px; margin: 12px 0; }}
      .broker-summary-label {{ color: {SECONDARY_TEXT}; font-size: 12px; }}
      .broker-summary-value {{ color: {TEXT}; font-size: 22px; font-weight: 750; margin-top: 3px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

broker_col1, broker_col2 = st.columns([0.92, 1.08], gap="large")

with broker_col1:
    with st.container(border=True):
        st.markdown('<div class="broker-eyebrow">01 · ACCOUNT PROVIDER</div>', unsafe_allow_html=True)
        st.markdown('<div class="broker-title">Broker connection</div>', unsafe_allow_html=True)
        st.markdown('<div class="broker-muted">Choose a provider to view its official API resources and integration status.</div>', unsafe_allow_html=True)
        broker = st.selectbox(
            "Broker provider",
            ["Zerodha", "Upstox", "Other / Future Integration"],
            key="broker_provider",
            label_visibility="collapsed",
        )

        if broker == "Zerodha":
            broker_url = "https://kite.trade/"
            broker_docs = "Kite Connect"
            broker_description = "Review Zerodha’s official Kite Connect platform for API access, authentication requirements and developer documentation."
            broker_status = "API AVAILABLE · NOT CONNECTED"
        elif broker == "Upstox":
            broker_url = "https://upstox.com/developer/api-documentation/"
            broker_docs = "Upstox Developer API"
            broker_description = "Review Upstox’s official documentation for OAuth authentication, market data and order workflow requirements."
            broker_status = "API AVAILABLE · NOT CONNECTED"
        else:
            broker_url = None
            broker_docs = "Broker integration"
            broker_description = "Add a provider only after reviewing its official API, authentication flow, permissions and order rules."
            broker_status = "NOT CONFIGURED"

        st.markdown(f'<div class="broker-pill">● {broker_status}</div>', unsafe_allow_html=True)
        st.write(broker_description)
        if broker_url:
            st.link_button(f"Open {broker_docs} ↗", broker_url, use_container_width=True, type="primary")
        st.divider()
        st.markdown("**Security checklist**")
        st.markdown(
            "- Authenticate through the broker’s official flow.\n"
            "- Keep API secrets and access tokens on a secure backend.\n"
            "- Request only the permissions the app needs.\n"
            "- Require explicit confirmation before any live order is submitted."
        )

with broker_col2:
    with st.container(border=True):
        st.markdown('<div class="broker-eyebrow">02 · TRADE DETAILS</div>', unsafe_allow_html=True)
        st.markdown('<div class="broker-title">Order ticket</div>', unsafe_allow_html=True)
        st.markdown('<div class="broker-muted">Build a review-only ticket using the selected stock and latest downloaded closing price.</div>', unsafe_allow_html=True)

        order_side = st.radio("Transaction", ["BUY", "SELL"], horizontal=True, key="order_side")
        order_col1, order_col2 = st.columns(2)
        with order_col1:
            order_quantity = st.number_input("Quantity (shares)", min_value=1, value=1, step=1, key="order_quantity")
        with order_col2:
            order_type = st.selectbox("Order type", ["MARKET", "LIMIT"], key="order_type")

        product_type = st.selectbox(
            "Product type",
            ["CNC · Delivery", "MIS · Intraday (broker rules apply)"],
            key="order_product_type",
        )

        if order_type == "LIMIT":
            limit_price = st.number_input(
                "Limit price (₹)", min_value=0.01,
                value=max(0.01, float(round(latest_price, 2))),
                step=0.05, format="%.2f", key="limit_price",
            )
            estimated_unit_price = float(limit_price)
        else:
            limit_price = 0.0
            estimated_unit_price = float(latest_price)
            st.caption("Market orders may fill at a price different from the displayed reference price.")

        estimated_value = estimated_unit_price * int(order_quantity)
        summary_col1, summary_col2 = st.columns(2)
        summary_col1.metric("Selected stock", company_name)
        summary_col2.metric("Reference price", f"₹{latest_price:,.2f}")
        st.markdown(
            f'<div class="broker-summary"><div class="broker-summary-label">Estimated order value · before charges</div><div class="broker-summary-value">₹{estimated_value:,.2f}</div><div class="broker-muted">{order_side} · {stock} · {int(order_quantity):,} share(s)</div></div>',
            unsafe_allow_html=True,
        )
        st.caption("Estimate only. Taxes, brokerage, slippage, margin rules and actual execution price are not included.")

        review_acknowledged = st.checkbox(
            "I understand this creates a preview only and will not submit a live order.",
            key="order_preview_acknowledged",
        )
        if st.button("Prepare order preview", use_container_width=True, type="primary", disabled=not review_acknowledged, key="prepare_broker_order"):
            st.session_state.order_review_payload = {
                "broker": broker,
                "exchange": "NSE",
                "symbol": stock,
                "company": company_name,
                "transaction_type": order_side,
                "quantity": int(order_quantity),
                "product_type": product_type,
                "order_type": order_type,
                "price": float(limit_price),
                "reference_market_price": float(latest_price),
                "estimated_value_before_charges": round(float(estimated_value), 2),
                "forecast_signal_for_context_only": signal_label,
                "decision_score_for_context_only": round(float(signal_score), 2),
                "live_submission": False,
            }

        if st.session_state.get("order_review_payload"):
            import json as _json
            st.divider()
            st.markdown("**Prepared preview**")
            st.success("Preview created. No broker account was accessed and no order was sent.")
            st.code(_json.dumps(st.session_state.order_review_payload, indent=2), language="json")

st.info(
    "Live trading is not enabled in this version. A real connection requires broker-approved authentication, "
    "server-side credential handling, broker-specific instrument identifiers and explicit order submission controls.",
    icon="🔒",
)


# ============================================================
# FUTURE DATES
# ============================================================

last_date = (
    data["Date"].iloc[-1]
)


future_dates = pd.bdate_range(
    start=(
        last_date +
        pd.Timedelta(days=1)
    ),
    periods=prediction_days
)


# ============================================================
# AI STOCK FORECAST
# ============================================================

st.markdown('<div id="ai-stock-forecast"></div>', unsafe_allow_html=True)

st.subheader(
    "📈 AI Stock Forecast"
)


st.caption(
    "Historical market price followed by independent "
    "LSTM and ARIMA future forecasts."
)


forecast_chart = go.Figure()


# ------------------------------------------------------------
# HISTORICAL PRICE
# ------------------------------------------------------------

forecast_chart.add_trace(
    go.Scatter(
        x=data["Date"],
        y=data["Close"],
        mode="lines",
        name="Historical Price",
        line=dict(
            width=2.5
        ),
        hovertemplate=
            "<b>Date:</b> %{x|%d %b %Y}"
            "<br>"
            "<b>Price:</b> ₹%{y:,.2f}"
            "<extra></extra>"
    )
)


# ------------------------------------------------------------
# LSTM FUTURE
# ------------------------------------------------------------

forecast_chart.add_trace(
    go.Scatter(
        x=future_dates,
        y=future_lstm,
        mode="lines",
        name="LSTM Forecast",
        line=dict(
            width=3
        ),
        hovertemplate=
            "<b>Date:</b> %{x|%d %b %Y}"
            "<br>"
            "<b>LSTM:</b> ₹%{y:,.2f}"
            "<extra></extra>"
    )
)


# ------------------------------------------------------------
# ARIMA FUTURE
# ------------------------------------------------------------

forecast_chart.add_trace(
    go.Scatter(
        x=future_dates,
        y=arima_forecast,
        mode="lines",
        name="ARIMA Forecast",
        line=dict(
            width=3,
            dash="dash"
        ),
        hovertemplate=
            "<b>Date:</b> %{x|%d %b %Y}"
            "<br>"
            "<b>ARIMA:</b> ₹%{y:,.2f}"
            "<extra></extra>"
    )
)


# ------------------------------------------------------------
# FORECAST START MARKER
# ------------------------------------------------------------

forecast_chart.add_vline(
    x=last_date,
    line_width=2,
    line_dash="dot"
)


forecast_chart.add_annotation(
    x=last_date,
    y=1,
    yref="paper",
    text="Forecast Start",
    showarrow=False,
    yshift=14,
    font=dict(
        size=12
    )
)


# ------------------------------------------------------------
# FORECAST AREA
# ------------------------------------------------------------

forecast_chart.add_vrect(
    x0=future_dates[0],
    x1=future_dates[-1],
    opacity=0.04,
    line_width=0
)


forecast_chart.update_layout(
    template=PLOT_TEMPLATE,
    height=650,
    title="Historical Price & Future Forecast",
    xaxis_title="Date",
    yaxis_title="Price",
    paper_bgcolor=BACKGROUND,
    plot_bgcolor=("#0E1117" if dark_mode else "#FFFFFF"),
    font=dict(color=TEXT),
    hovermode="x unified",
    margin=dict(
        l=60,
        r=30,
        t=100,
        b=60
    )
)

forecast_chart = apply_plotly_theme(forecast_chart)

st.plotly_chart(
    forecast_chart,
    use_container_width=True,
    config={
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": True
    }
)


# ============================================================
# FUTURE PREDICTION TABLE
# ============================================================

st.markdown('<div id="future-predictions"></div>', unsafe_allow_html=True)

st.subheader(
    "📋 Future Predictions"
)


future_predictions = pd.DataFrame(
    {
        "Date": future_dates,
        "LSTM Prediction": future_lstm,
        "ARIMA Prediction": arima_forecast
    }
)


future_predictions[
    "LSTM Prediction"
] = future_predictions[
    "LSTM Prediction"
].round(2)


future_predictions[
    "ARIMA Prediction"
] = future_predictions[
    "ARIMA Prediction"
].round(2)


# Add expected movement percentages

future_predictions[
    "LSTM Change %"
] = (
    (
        future_predictions[
            "LSTM Prediction"
        ] -
        latest_price
    )
    /
    latest_price
    * 100
).round(2)


future_predictions[
    "ARIMA Change %"
] = (
    (
        future_predictions[
            "ARIMA Prediction"
        ] -
        latest_price
    )
    /
    latest_price
    * 100
).round(2)


st.dataframe(
    future_predictions,
    use_container_width=True,
    height=420
)


# ============================================================
# FINAL FORECAST SUMMARY
# ============================================================
# IMPORTANT:
# This section intentionally appears AFTER the table.
# ============================================================

lstm_final = float(
    future_lstm[-1]
)


arima_final = float(
    arima_forecast[-1]
)


lstm_change = (
    (
        lstm_final -
        latest_price
    )
    /
    latest_price
) * 100


arima_change = (
    (
        arima_final -
        latest_price
    )
    /
    latest_price
) * 100


st.subheader(
    "🎯 Forecast Summary"
)


summary_col1, summary_col2, summary_col3 = (
    st.columns(3)
)


summary_col1.metric(
    "Current Price",
    f"₹{latest_price:,.2f}"
)


summary_col2.metric(
    f"LSTM ({prediction_days} Days)",
    f"₹{lstm_final:,.2f}",
    f"{lstm_change:+.2f}%"
)


summary_col3.metric(
    f"ARIMA ({prediction_days} Days)",
    f"₹{arima_final:,.2f}",
    f"{arima_change:+.2f}%"
)


# ============================================================
# MODEL INTERPRETATION
# ============================================================

with st.expander(
    "ℹ️ About the Prediction Models"
):

    st.markdown(
        """
        **LSTM — Long Short-Term Memory**

        The LSTM neural network is used to learn sequential
        relationships in historical stock-price data and
        recursively generate future price estimates.

        **ARIMA — Autoregressive Integrated Moving Average**

        ARIMA provides an independent statistical time-series
        forecast based on historical price behaviour.

        **Model Evaluation**

        RMSE, MAE and MAPE are calculated on the historical validation
        period for both LSTM and ARIMA where ARIMA validation converges.
        These metrics are used by the StockAI Decision Engine as historical
        context, not as a guarantee of future accuracy.

        **Important**

        The LSTM and ARIMA forecasts are independent model
        estimates. They should not be interpreted as guaranteed
        future market prices.
        """
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()


st.warning(
    "⚠️ Disclaimer: StockAI provides model-based "
    "analysis and decision-support information for "
    "educational and research purposes. Stock prices are "
    "affected by many unpredictable factors. Forecasts, "
    "scores and outlook labels are not guaranteed and should "
    "not be treated as personalized financial or investment advice. "
    "Live broker execution is not performed by this prototype."
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "StockAI | LSTM + ARIMA | AI-Based Stock Market "
    "Analysis and Forecasting System"
)