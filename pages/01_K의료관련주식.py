import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="한국 의료관련주식 전문 분석",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 한국 의료관련주식 전문 분석 by Suhee Agnes")
st.caption("Yahoo Finance 기반 한국 헬스케어·제약·의료기기·AI의료 종목 비교 대시보드")

KOREA_HEALTHCARE_STOCKS = {
    "삼성바이오로직스": "207940.KS",
    "셀트리온": "068270.KS",
    "유한양행": "000100.KS",
    "한미약품": "128940.KS",
    "녹십자": "006280.KS",
    "종근당": "185750.KS",
    "대웅제약": "069620.KS",
    "HK이노엔": "195940.KS",
    "씨젠": "096530.KQ",
    "루닛": "328130.KQ",
    "클래시스": "214150.KQ",
    "덴티움": "145720.KQ",
    "파마리서치": "214450.KQ",
    "아이센스": "099190.KQ",
    "바텍": "043150.KQ",
}

CATEGORY_MAP = {
    "삼성바이오로직스": "제약·바이오",
    "셀트리온": "제약·바이오",
    "유한양행": "제약·바이오",
    "한미약품": "제약·바이오",
    "녹십자": "제약·바이오",
    "종근당": "제약·바이오",
    "대웅제약": "제약·바이오",
    "HK이노엔": "제약·바이오",
    "씨젠": "진단·분자진단",
    "루닛": "AI의료",
    "클래시스": "의료기기·미용의료",
    "덴티움": "의료기기·치과",
    "파마리서치": "헬스케어·재생의학",
    "아이센스": "체외진단·센서",
    "바텍": "의료기기·영상장비",
}

BENCHMARKS = {
    "KODEX 헬스케어": "266420.KS",
    "KOSPI": "^KS11",
    "KOSDAQ": "^KQ11",
}

@st.cache_data(ttl=3600)
def load_multi_data(tickers, period="1y", interval="1d"):
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

def extract_series(raw_data, ticker, field="Close"):
    if isinstance(raw_data.columns, pd.MultiIndex):
        if ticker in raw_data.columns.get_level_values(0):
            df = raw_data[ticker].copy()
        else:
            df = raw_data.copy()
    else:
        df = raw_data.copy()

    if field in df.columns:
        s = df[field].copy()
    else:
        s = pd.Series(index=df.index, dtype="float64")

    s.name = ticker
    return s

def build_field_dataframe(raw_data, tickers, field="Close"):
    frames = []
    for ticker in tickers:
        try:
            s = extract_series(raw_data, ticker, field=field)
            frames.append(s)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1)
    df = df.dropna(how="all")
    df = df.sort_index().ffill()
    return df

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stock_score(close_series, benchmark_series):
    close_series = close_series.dropna()
    benchmark_series = benchmark_series.dropna()

    if len(close_series) < 70 or len(benchmark_series) < 70:
        return None

    returns = close_series.pct_change().dropna()
    benchmark_returns = benchmark_series.pct_change().dropna()

    common_idx = returns.index.intersection(benchmark_returns.index)
    returns = returns.loc[common_idx]
    benchmark_returns = benchmark_returns.loc[common_idx]

    if len(common_idx) < 40:
        return None

    last_price = close_series.iloc[-1]
    ma20 = close_series.rolling(20).mean().iloc[-1]
    ma60 = close_series.rolling(60).mean().iloc[-1]

    ret_1m = ((close_series.iloc[-1] / close_series.iloc[-21]) - 1) * 100 if len(close_series) > 21 else np.nan
    ret_3m = ((close_series.iloc[-1] / close_series.iloc[-63]) - 1) * 100 if len(close_series) > 63 else np.nan

    vol_1m = returns.tail(21).std() * np.sqrt(252) * 100 if len(returns) >= 21 else np.nan
    rsi_14 = compute_rsi(close_series, 14).iloc[-1]

    rel_3m = (
        ((close_series.iloc[-1] / close_series.iloc[-63]) - 1) -
        ((benchmark_series.iloc[-1] / benchmark_series.iloc[-63]) - 1)
    ) * 100 if len(close_series) > 63 and len(benchmark_series) > 63 else np.nan

    trend_score = 0
    if pd.notna(ma20) and pd.notna(ma60):
        if last_price > ma20:
            trend_score += 1
        if last_price > ma60:
            trend_score += 1
        if ma20 > ma60:
            trend_score += 1

    momentum_score = 0
    if pd.notna(ret_1m) and ret_1m > 0:
        momentum_score += 1
    if pd.notna(ret_3m) and ret_3m > 0:
        momentum_score += 1
    if pd.notna(rel_3m) and rel_3m > 0:
        momentum_score += 1

    rsi_score = 0
    if pd.notna(rsi_14):
        if 50 <= rsi_14 <= 70:
            rsi_score = 2
        elif 40 <= rsi_14 < 50 or 70 < rsi_14 <= 80:
            rsi_score = 1

    risk_penalty = 0
    if pd.notna(vol_1m):
        if vol_1m > 70:
            risk_penalty = -2
        elif vol_1m > 50:
            risk_penalty = -1

    total_score = trend_score + momentum_score + rsi_score + risk_penalty

    return {
        "최근가격": last_price,
        "1개월수익률(%)": ret_1m,
        "3개월수익률(%)": ret_3m,
        "연환산변동성(1M, %)": vol_1m,
        "RSI(14)": rsi_14,
        "3개월상대강도(%)": rel_3m,
        "추세점수": trend_score,
        "모멘텀점수": momentum_score,
        "RSI점수": rsi_score,
        "리스크보정": risk_penalty,
        "종합점수": total_score,
    }

with st.sidebar:
    st.header("분석 설정")

    selected_categories = st.multiselect(
        "세부 분야 선택",
        options=sorted(list(set(CATEGORY_MAP.values()))),
        default=sorted(list(set(CATEGORY_MAP.values()))),
    )

    available_names = [
        name for name, category in CATEGORY_MAP.items()
        if category in selected_categories
    ]

    selected_names = st.multiselect(
        "종목 선택",
        options=available_names,
        default=available_names[:8] if len(available_names) >= 8 else available_names,
    )

    period = st.selectbox(
        "조회 기간",
        ["3mo", "6mo", "1y", "2y", "5y"],
        index=2,
    )

    benchmark_name = st.selectbox(
        "비교 벤치마크",
        options=list(BENCHMARKS.keys()),
        index=0,
    )

    normalize_chart = st.toggle("상대성과 비교용 정규화 차트", value=True)

if not selected_names:
    st.warning("최소 1개 이상의 종목을 선택하세요.")
    st.stop()

selected_tickers = [KOREA_HEALTHCARE_STOCKS[name] for name in selected_names]
benchmark_ticker = BENCHMARKS[benchmark_name]
all_tickers = selected_tickers + [benchmark_ticker]

with st.spinner("한국 의료관련주 데이터를 불러오는 중..."):
    raw = load_multi_data(all_tickers, period=period)

close_df = build_field_dataframe(raw, all_tickers, field="Close")
volume_df = build_field_dataframe(raw, all_tickers, field="Volume")

if close_df.empty:
    st.error("데이터를 불러오지 못했습니다. 종목 코드나 기간을 변경해 보세요.")
    st.stop()

rename_map = {v: k for k, v in KOREA_HEALTHCARE_STOCKS.items()}
rename_map[benchmark_ticker] = benchmark_name

close_df = close_df.rename(columns=rename_map)
volume_df = volume_df.rename(columns=rename_map)

stock_columns = [name for name in selected_names if name in close_df.columns]
if len(stock_columns) == 0 or benchmark_name not in close_df.columns:
    st.error("분석 가능한 데이터가 부족합니다.")
    st.stop()

benchmark_series = close_df[benchmark_name]
latest_prices = close_df[stock_columns].iloc[-1]
prev_prices = close_df[stock_columns].iloc[-2] if len(close_df) > 1 else latest_prices
daily_changes = ((latest_prices / prev_prices) - 1) * 100

st.subheader("핵심 지표")
metric_cols = st.columns(min(4, len(stock_columns)))
leaders = daily_changes.sort_values(ascending=False)

for i, name in enumerate(leaders.index[:min(4, len(stock_columns))]):
    metric_cols[i].metric(
        label=name,
        value=f"{latest_prices[name]:,.0f}",
        delta=f"{daily_changes[name]:+.2f}%",
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["종합 스코어", "가격 비교", "상대강도", "리스크", "개별 심층분석"]
)

with tab1:
    score_rows = []
    for stock_name in stock_columns:
        result = calculate_stock_score(close_df[stock_name], benchmark_series)
        if result is not None:
            result["종목"] = stock_name
            result["세부분야"] = CATEGORY_MAP.get(stock_name, "기타")
            score_rows.append(result)

    if not score_rows:
        st.info("점수를 계산할 데이터가 충분하지 않습니다.")
    else:
        score_df = pd.DataFrame(score_rows)
        score_df = score_df[
            [
                "종목", "세부분야", "최근가격", "1개월수익률(%)", "3개월수익률(%)",
                "연환산변동성(1M, %)", "RSI(14)", "3개월상대강도(%)",
                "추세점수", "모멘텀점수", "RSI점수", "리스크보정", "종합점수"
            ]
        ].sort_values("종합점수", ascending=False)

        st.dataframe(
            score_df.style.format({
                "최근가격": "{:,.0f}",
                "1개월수익률(%)": "{:+.2f}",
                "3개월수익률(%)": "{:+.2f}",
                "연환산변동성(1M, %)": "{:.2f}",
                "RSI(14)": "{:.2f}",
                "3개월상대강도(%)": "{:+.2f}",
            }),
            use_container_width=True,
        )

        fig_score = px.bar(
            score_df,
            x="종목",
            y="종합점수",
            color="세부분야",
            text="종합점수",
            title="한국 의료관련주 종합 스코어",
        )
        fig_score.update_layout(
            height=500,
            template="plotly_white",
            xaxis_title="종목",
            yaxis_title="종합점수",
        )
        st.plotly_chart(fig_score, use_container_width=True)

        leader = score_df.iloc[0]
        st.success(
            f"현재 종합 스코어 1위는 {leader['종목']}이며, "
            f"세부분야는 {leader['세부분야']}, "
            f"3개월 수익률은 {leader['3개월수익률(%)']:+.2f}%입니다."
        )

with tab2:
    price_df = close_df[stock_columns].copy()

    if normalize_chart:
        price_df = (price_df / price_df.iloc[0]) * 100
        y_title = "Normalized Price (Start=100)"
        chart_title = "한국 의료관련주 상대 성과 비교"
    else:
        y_title = "Price"
        chart_title = "한국 의료관련주 절대 가격 비교"

    fig_price = px.line(
        price_df,
        x=price_df.index,
        y=price_df.columns,
        title=chart_title,
        labels={"value": y_title, "index": "Date", "variable": "종목"},
    )
    fig_price.update_layout(
        height=520,
        hovermode="x unified",
        template="plotly_white",
        legend_title_text="종목",
    )
    fig_price.update_traces(line=dict(width=2))
    st.plotly_chart(fig_price, use_container_width=True)

    cum_return = ((close_df[stock_columns].iloc[-1] / close_df[stock_columns].iloc[0]) - 1) * 100
    perf_df = pd.DataFrame({
        "종목": cum_return.index,
        "세부분야": [CATEGORY_MAP.get(x, "기타") for x in cum_return.index],
        "누적수익률(%)": cum_return.values,
        "최근가격": latest_prices[cum_return.index].values,
        "일간변동률(%)": daily_changes[cum_return.index].values,
    }).sort_values("누적수익률(%)", ascending=False)

    st.dataframe(
        perf_df.style.format({
            "누적수익률(%)": "{:+.2f}",
            "최근가격": "{:,.0f}",
            "일간변동률(%)": "{:+.2f}",
        }),
        use_container_width=True,
    )

with tab3:
    rel_df = pd.DataFrame(index=close_df.index)

    for stock_name in stock_columns:
        rel_df[stock_name] = (
            (close_df[stock_name] / close_df[stock_name].iloc[0]) /
            (benchmark_series / benchmark_series.iloc[0])
        ) * 100

    fig_rel = px.line(
        rel_df,
        x=rel_df.index,
        y=rel_df.columns,
        title=f"{benchmark_name} 대비 상대강도 비교 (시작값=100)",
        labels={"value": "Relative Strength", "index": "Date", "variable": "종목"},
    )
    fig_rel.update_layout(
        height=520,
        hovermode="x unified",
        template="plotly_white",
    )
    fig_rel.add_hline(y=100, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_rel, use_container_width=True)

    latest_rel = rel_df.iloc[-1].sort_values(ascending=False)
    rel_rank_df = pd.DataFrame({
        "종목": latest_rel.index,
        "세부분야": [CATEGORY_MAP.get(x, "기타") for x in latest_rel.index],
        "상대강도지수": latest_rel.values,
    })
    st.dataframe(
        rel_rank_df.style.format({"상대강도지수": "{:.2f}"}),
        use_container_width=True,
    )

with tab4:
    returns = close_df[stock_columns].pct_change().dropna()
    vol_df = returns.rolling(21).std() * np.sqrt(252) * 100

    if not vol_df.empty:
        fig_vol = px.line(
            vol_df,
            x=vol_df.index,
            y=vol_df.columns,
            title="연환산 변동성 추이 (21일 기준)",
            labels={"value": "Volatility (%)", "index": "Date", "variable": "종목"},
        )
        fig_vol.update_layout(
            height=520,
            hovermode="x unified",
            template="plotly_white",
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    corr_df = returns.tail(60).corr()
    fig_corr = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="최근 60거래일 수익률 상관행렬",
    )
    fig_corr.update_layout(height=520, template="plotly_white")
    st.plotly_chart(fig_corr, use_container_width=True)

with tab5:
    detail_stock = st.selectbox("심층분석 종목", options=stock_columns, index=0)

    detail_df = pd.DataFrame({
        "Close": close_df[detail_stock]
    }).dropna()

    if detail_stock in volume_df.columns:
        detail_df["Volume"] = volume_df[detail_stock]

    detail_df["MA20"] = detail_df["Close"].rolling(20).mean()
    detail_df["MA60"] = detail_df["Close"].rolling(60).mean()
    detail_df["RSI14"] = compute_rsi(detail_df["Close"], 14)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("최근 가격", f"{detail_df['Close'].iloc[-1]:,.0f}")
    d2.metric(
        "1개월 수익률",
        f"{((detail_df['Close'].iloc[-1] / detail_df['Close'].iloc[-21]) - 1) * 100:+.2f}%"
        if len(detail_df) > 21 else "N/A"
    )
    d3.metric(
        "3개월 수익률",
        f"{((detail_df['Close'].iloc[-1] / detail_df['Close'].iloc[-63]) - 1) * 100:+.2f}%"
        if len(detail_df) > 63 else "N/A"
    )
    d4.metric(
        "RSI(14)",
        f"{detail_df['RSI14'].iloc[-1]:.2f}" if pd.notna(detail_df["RSI14"].iloc[-1]) else "N/A"
    )

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
            y=detail_df["MA20"],
            mode="lines",
            name="MA20",
            line=dict(dash="dot"),
        )
    )
    fig_detail.add_trace(
        go.Scatter(
            x=detail_df.index,
            y=detail_df["MA60"],
            mode="lines",
            name="MA60",
            line=dict(dash="dash"),
        )
    )
    fig_detail.update_layout(
        title=f"{detail_stock} 가격 및 이동평균선",
        height=520,
        hovermode="x unified",
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Price",
    )
    st.plotly_chart(fig_detail, use_container_width=True)

    fig_rsi = go.Figure()
    fig_rsi.add_trace(
        go.Scatter(
            x=detail_df.index,
            y=detail_df["RSI14"],
            mode="lines",
            name="RSI(14)",
            line=dict(width=2, color="#00897B"),
        )
    )
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="blue")
    fig_rsi.update_layout(
        title=f"{detail_stock} RSI(14)",
        height=320,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="RSI",
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

    if "Volume" in detail_df.columns and detail_df["Volume"].notna().any():
        fig_volume = px.bar(
            detail_df.reset_index(),
            x="Date",
            y="Volume",
            title=f"{detail_stock} 거래량",
            template="plotly_white",
        )
        fig_volume.update_layout(height=300)
        st.plotly_chart(fig_volume, use_container_width=True)

st.markdown("---")
st.markdown(
    """
    **해석 기준**
    - 종합점수: 추세, 모멘텀, RSI, 변동성을 단순 결합한 내부 참고용 지표
    - 상대강도: 선택한 벤치마크 대비 outperform 여부 확인용
    - RSI: 일반적으로 70 이상은 과열, 30 이하는 과매도 구간으로 자주 해석
    """
)
st.caption("Data source: Yahoo Finance via yfinance")
