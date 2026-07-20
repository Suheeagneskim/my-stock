import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Global Market Dashboard",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Global Market Stock Dashboard")
st.caption("Yahoo Finance 기반 글로벌 주요 지수 및 대표 종목 대시보드")

MARKETS = {
    "미국": {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "NVIDIA": "NVDA",
        "Apple": "AAPL",
        "Microsoft": "MSFT",
    },
    "유럽": {
        "FTSE 100": "^FTSE",
        "DAX": "^GDAXI",
        "CAC 40": "^FCHI",
        "ASML": "ASML",
        "SAP": "SAP",
        "LVMH": "MC.PA",
    },
    "아시아": 
