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

st.title("🌍 Global Market Stock Dashboard by Suhee Agnes")
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
    "아시아": {
        "Nikkei 225": "^N225",
        "Hang Seng": "^HSI",
        "KOSPI": "^KS11",
        "Samsung Electronics": "005930.KS",
        "Toyota": "7203.T",
        "TSMC": "TSM",
    },
}

ALL_OPTIONS = {}
for region, items in MARKETS.items():
    for name, ticker in items.items():
        ALL_OPTIONS[f"{region} - {name} ({ticker})"] = ticker

@st.cache_data(ttl=3600)
def load_price_data(tickers, period="1y", interval="1d"):
    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    return data

@st.cache_data(ttl=3600)
def load_recent_history(ticker):
    t = yf.Ticker(ticker)
    hist = t.history(period="6mo", auto_adjust=True)
    return hist

def extract_close_series(raw_data, ticker):
    if isinstance(raw_data.columns, pd.MultiIndex):
        if ticker in raw_data.columns.get_level_values(0):
            df = raw_data[ticker].copy()
        else:
            df = raw_data.copy()
    else:
        df = raw_data.copy()

    if "Close" in df.columns:
        s = df["Close"].copy()
    else:
        s = df.squeeze().copy()

    s.name = ticker
    return s

def build_close_dataframe(raw_data, tickers):
    frames = []
    for ticker in tickers:
        try:
            s = extract_close_series(raw_data, ticker)
            frames.append(s)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    close_df = pd.concat(frames, axis=1)
    close_df = close_df.dropna(how="all")
    return close_df

with st.sidebar:
    st.header("설정")
    selected_assets = st.multiselect(
        "종목/지수 선택",
        options=list(ALL_OPTIONS.keys()),
        default=[
            "미국 - S&P 500 (^GSPC)",
            "미국 - NASDAQ (^IXIC)",
            "유럽 - DAX (^GDAXI)",
            "아시아 - Nikkei 225 (^N225)",
            "아시아 - KOSPI (^KS11)",
        ],
    )

    period = st.selectbox(
        "조회 기간",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3,
    )

    chart_mode = st.radio(
        "차트 표시 방식",
        options=["정규화 비교", "절대 가격"],
        index=0,
    )

if not selected_assets:
    st.warning("왼쪽에서 최소 1개 이상의 종목 또는 지수를 선택하세요.")
    st.stop()

selected_tickers = [ALL_OPTIONS[label] for label in selected_assets]
label_map = {v: k.split(" - ")[1].rsplit(" (", 1)[0] for k, v in ALL_OPTIONS.items()}

with st.spinner("Yahoo Finance 데이터 불러오는 중..."):
    raw = load_price_data(selected_tickers, period=period)
    close_df = build_close_dataframe(raw, selected_tickers)

if close_df.empty:
    st.error("데이터를 불러오지 못했습니다. 다른 종목이나 기간으로 다시 시도하세요.")
    st.stop()

close_df = close_df.rename(columns=label_map)
close_df = close_df.sort_index().ffill()

if len(close_df) < 2:
    st.error("비교 가능한 데이터가 충분하지 않습니다.")
    st.stop()

latest = close_df.iloc[-1]
prev = close_df.iloc[-2]
daily_change = ((latest / prev) - 1) * 100
period_return = ((close_df.iloc[-1] / close_df.iloc[0]) - 1) * 100

st.subheader("시장 요약")

metric_count = min(4, len(close_df.columns))
metric_cols = st.columns(metric_count)
for i, col_name in enumerate(close_df.columns[:metric_count]):
    metric_cols[i].metric(
        label=col_name,
        value=f"{latest[col_name]:,.2f}",
        delta=f"{daily_change[col_name]:+.2f}%",
    )

tab1, tab2, tab3 = st.tabs(["가격 추이", "일간 수익률", "개별 종목 상세"])

with tab1:
    plot_df = close_df.copy()

    if chart_mode == "정규화 비교":
        plot_df = (plot_df / plot_df.iloc[0]) * 100
        y_title = "Normalized Price (Start=100)"
        chart_title = "주요 시장 성과 비교"
    else:
        y_title = "Price"
        chart_title = "주요 시장 절대 가격 추이"

    fig = px.line(
        plot_df,
        x=plot_df.index,
        y=plot_df.columns,
        labels={"value": y_title, "index": "Date", "variable": "Asset"},
        title=chart_title,
    )
    fig.update_layout(
        hovermode="x unified",
        legend_title_text="자산",
        height=520,
        template="plotly_white",
    )
    fig.update_traces(line=dict(width=2))
    st.plotly_chart(fig, use_container_width=True)

    ranking = pd.DataFrame({
        "자산": period_return.index,
        "기간 수익률(%)": period_return.values,
        "최근 가격": latest.values,
        "일간 변동률(%)": daily_change.values,
    }).sort_values("기간 수익률(%)", ascending=False)

    st.dataframe(
        ranking.style.format({
            "기간 수익률(%)": "{:+.2f}",
            "최근 가격": "{:,.2f}",
            "일간 변동률(%)": "{:+.2f}",
        }),
        use_container_width=True,
    )

with tab2:
    returns = close_df.pct_change() * 100
    heatmap_df = returns.tail(30).T

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=heatmap_df.values,
            x=[d.strftime("%Y-%m-%d") for d in heatmap_df.columns],
            y=heatmap_df.index,
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="%"),
            hovertemplate="자산: %{y}<br>날짜: %{x}<br>수익률: %{z:.2f}%<extra></extra>",
        )
    )
    fig_heat.update_layout(
        title="최근 30거래일 일간 수익률 히트맵",
        height=520,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Asset",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    selected_detail = st.selectbox(
        "상세 조회 자산 선택",
        options=list(close_df.columns),
        index=0,
    )

    original_ticker = None
    for ticker, display_name in label_map.items():
        if display_name == selected_detail:
            original_ticker = ticker
            break

    if original_ticker:
        hist = load_recent_history(original_ticker)

        if hist.empty:
            st.warning("해당 자산의 상세 데이터를 불러오지 못했습니다.")
        else:
            hist = hist.sort_index().copy()

            last_price = hist["Close"].iloc[-1] if "Close" in hist.columns else None
            open_price = hist["Open"].iloc[-1] if "Open" in hist.columns else None
            high_price = hist["High"].iloc[-1] if "High" in hist.columns else None
            low_price = hist["Low"].iloc[-1] if "Low" in hist.columns else None

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("최근 종가", f"{last_price:,.2f}" if last_price is not None else "N/A")
            c2.metric("당일 시가", f"{open_price:,.2f}" if open_price is not None else "N/A")
            c3.metric("당일 고가", f"{high_price:,.2f}" if high_price is not None else "N/A")
            c4.metric("당일 저가", f"{low_price:,.2f}" if low_price is not None else "N/A")

            detail_df = hist[["Close"]].copy()
            detail_df["20D_MA"] = detail_df["Close"].rolling(20).mean()
            detail_df["60D_MA"] = detail_df["Close"].rolling(60).mean()

            fig_detail = go.Figure()
            fig_detail.add_trace(
                go.Scatter(
                    x=detail_df.index,
                    y=detail_df["Close"],
                    mode="lines",
                    name="Close",
                    line=dict(width=2),
                )
            )
            fig_detail.add_trace(
                go.Scatter(
                    x=detail_df.index,
                    y=detail_df["20D_MA"],
                    mode="lines",
                    name="20D MA",
                    line=dict(dash="dot"),
                )
            )
            fig_detail.add_trace(
                go.Scatter(
                    x=detail_df.index,
                    y=detail_df["60D_MA"],
                    mode="lines",
                    name="60D MA",
                    line=dict(dash="dash"),
                )
            )
            fig_detail.update_layout(
                title=f"{selected_detail} 상세 차트",
                height=520,
                hovermode="x unified",
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Price",
            )
            st.plotly_chart(fig_detail, use_container_width=True)

            volume_available = "Volume" in hist.columns and hist["Volume"].notna().any()
            if volume_available:
                fig_volume = px.bar(
                    hist.reset_index(),
                    x="Date",
                    y="Volume",
                    title=f"{selected_detail} 거래량",
                    template="plotly_white",
                )
                fig_volume.update_layout(height=300)
                st.plotly_chart(fig_volume, use_container_width=True)

st.markdown("---")
st.caption("Data source: Yahoo Finance via yfinance")
