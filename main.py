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

# 최근 20년(끝 연도 포함 20개 연도 구간)만으로 구한 기울기
RECENT_YEARS = 20
recent_mask = x >= x.max() - (RECENT_YEARS - 1)
x_rec, y_rec = x[recent_mask], y[recent_mask]
has_recent = len(x_rec) >= 3
if has_recent:
    slope_rec, _ = np.polyfit(x_rec, y_rec, 1)
    r_rec = float(np.corrcoef(x_rec, y_rec)[0, 1])

# ---- 직선 정보 ----
c1, c2, c3 = st.columns(3)
c1.metric("직선을 만든 해의 개수", f"{len(yearly)}개")
c2.metric("시작 연도", f"{int(x.min())}년")
c3.metric("끝 연도", f"{int(x.max())}년")

st.metric("상관계수 (연도 vs 연평균기온)", f"{r:.3f}")

st.divider()
st.subheader("100년에 몇 도 오르는가?")


def big_slope_card(title: str, sub: str, value: float) -> str:
    return f"""
    <div style="text-align:center; padding:1rem 0.5rem; border:1px solid rgba(128,128,128,0.35);
                border-radius:12px;">
      <div style="font-size:1rem; font-weight:600;">{title}</div>
      <div style="font-size:0.85rem; opacity:0.65;">{sub}</div>
      <div style="font-size:3rem; font-weight:800; line-height:1.2;">{value:+.2f} ℃</div>
      <div style="font-size:0.85rem; opacity:0.65;">/ 100년</div>
    </div>
    """


col_all, col_rec = st.columns(2)
with col_all:
    st.markdown(
        big_slope_card(
            "전체 기간", f"{int(x.min())}~{int(x.max())}년 ({len(x)}개 해)", slope * 100
        ),
        unsafe_allow_html=True,
    )
with col_rec:
    if has_recent:
        st.markdown(
            big_slope_card(
                f"최근 {RECENT_YEARS}년",
                f"{int(x_rec.min())}~{int(x_rec.max())}년 ({len(x_rec)}개 해)",
                slope_rec * 100,
            ),
            unsafe_allow_html=True,
        )
    else:
        st.warning("최근 20년 구간에 해가 3개 미만이라 계산하지 않았습니다.")

if has_recent:
    diff = (slope_rec - slope) * 100
    ratio_txt = f" (전체 기간의 약 {slope_rec / slope:.1f}배)" if slope > 0 else ""
    st.markdown(
        f"최근 {RECENT_YEARS}년 기울기가 전체 기간보다 **{diff:+.2f} ℃ / 100년** 차이 나요{ratio_txt}. "
        f"상관계수는 전체 {r:.3f}, 최근 {RECENT_YEARS}년 {r_rec:.3f}입니다."
    )
    st.caption(
        f"최근 {RECENT_YEARS}년은 데이터 수가 적어 기울기가 해마다 크게 흔들릴 수 있습니다. "
        "짧은 구간의 값을 100년으로 환산한 것이라 그대로 미래 예측으로 받아들이면 안 돼요."
    )

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
