import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업 기준 기간의 마지막 해
MIN_DAYS = 300        # 연평균에 쓰려면 필요한 최소 관측일 수

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_yearly_mean() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]

    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜"])

    # 기준 기간(2025년까지)만 사용
    df = df[df["날짜"].dt.year <= LAST_YEAR].copy()
    df["연도"] = df["날짜"].dt.year

    # 평균기온이 실제로 관측된 날만 관측일로 센다
    obs = df.dropna(subset=["평균기온"])
    yearly = (
        obs.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    # 관측일이 300일 미만인 해는 제외
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS].reset_index(drop=True)
    return yearly


st.title("🌡️ 기온 예측기")
st.caption("서울 연평균기온으로 만든 회귀 직선으로 원하는 해의 기온을 예상해 봅니다.")

try:
    yearly = load_yearly_mean()
except Exception as e:  # 네트워크·형식 오류 안내
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 직선을 만들 수 있는 해가 부족합니다.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

# ---- 직선 정보 ----
c1, c2, c3 = st.columns(3)
c1.metric("직선을 만든 해의 개수", f"{len(yearly)}개")
c2.metric("시작 연도", f"{int(x.min())}년")
c3.metric("끝 연도", f"{int(x.max())}년")

c4, c5 = st.columns(2)
c4.metric("상관계수 (연도 vs 연평균기온)", f"{r:.3f}")
c5.metric("기울기", f"{slope * 10:+.2f} ℃ / 10년")

st.caption(
    f"사용 기준: {LAST_YEAR}년까지의 데이터, 관측일 {MIN_DAYS}일 이상인 해만 포함. "
    f"회귀식: 기온 = {slope:.4f} × 연도 + ({intercept:.2f})"
)

st.divider()

# ---- 슬라이더 & 예상 기온 ----
year = st.slider("예상할 연도", min_value=1900, max_value=2100, value=2030, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0;">
      <div style="font-size:1.1rem; opacity:0.7;">{year}년 예상 연평균기온</div>
      <div style="font-size:4.5rem; font-weight:800; line-height:1.1;">{pred:.2f} ℃</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- 그래프 ----
x_lo = min(x.min(), year)
x_hi = max(x.max(), year)
line_x = np.array([x_lo, x_hi])
line_y = slope * line_x + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=7, opacity=0.8),
        hovertemplate="%{x}년<br>%{y:.2f} ℃<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name=f"회귀 직선 (r = {r:.3f})",
        line=dict(width=3),
    )
)
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers+text",
        name=f"{year}년 예상",
        text=[f"{pred:.2f}℃"],
        textposition="top center",
        marker=dict(size=14, symbol="diamond", line=dict(width=2, color="white")),
        hovertemplate="%{x}년<br>예상 %{y:.2f} ℃<extra></extra>",
    )
)
fig.update_layout(
    title="서울 연평균기온과 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=10, r=10, t=80, b=10),
    height=460,
)
st.plotly_chart(fig, use_container_width=True)

if year < x.min() or year > x.max():
    st.info("선택한 연도가 자료 범위 밖이라, 직선을 그대로 연장한 값입니다.")
