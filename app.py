# -*- coding: utf-8 -*-
"""
개체군은 언제 진화하지 않을까?  -  하디-바인베르크 평형 탐구 웹앱
고등학교 생명과학 수업용 / Streamlit

학생은 조건 A~J(익명)를 조작하며 세대별 대립유전자 빈도와 유전자형 빈도를
관찰하고, 평형 유지에 필요한 5가지 조건을 스스로 찾아낸다.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =====================================================================
# 0. 페이지 설정
# =====================================================================
st.set_page_config(
    page_title="개체군은 언제 진화하지 않을까?",
    page_icon="🧬",
    layout="wide",
)

# =====================================================================
# 1. 조건 A~J 메타데이터 (내부 정답)
# =====================================================================
# is_eq=True 이면 하디-바인베르크 평형 조건, False 이면 함정(무관) 조건
CONDITIONS = {
    "A": {"real": "개체군 크기",        "is_eq": True,
          "why": "집단이 작으면 우연(유전적 부동)으로 대립유전자 빈도가 크게 변할 수 있다."},
    "B": {"real": "무작위 교배",        "is_eq": True,
          "why": "교배가 무작위가 아니면 대립유전자 빈도는 거의 그대로여도 유전자형 빈도가 p²:2pq:q²에서 벗어난다."},
    "C": {"real": "돌연변이 없음",      "is_eq": True,
          "why": "돌연변이는 한 대립유전자를 다른 대립유전자로 바꾸어 빈도를 변화시킨다."},
    "D": {"real": "이주(유전자 흐름) 없음", "is_eq": True,
          "why": "외부 개체가 들어오면 다른 대립유전자 비율이 섞여 빈도가 변한다."},
    "E": {"real": "자연선택 없음",      "is_eq": True,
          "why": "유전자형마다 생존율이 다르면 특정 대립유전자가 늘거나 줄어든다."},
    "F": {"real": "초기 A 대립유전자 비율", "is_eq": False,
          "why": "시작값(p, q)과 p²:2pq:q² 값은 바뀌지만, 평형이 '유지되는지'와는 무관하다."},
    "G": {"real": "관찰 세대 수",       "is_eq": False,
          "why": "오래 관찰한다고 평형이 깨지거나 만들어지지 않는다. 관찰 기간일 뿐이다."},
    "H": {"real": "개체 이동 속도",     "is_eq": False,
          "why": "화면에서 점이 움직이는 속도일 뿐, 유전자 빈도에는 영향이 없다."},
    "I": {"real": "배경 환경(서식지 색)", "is_eq": False,
          "why": "생존율을 바꾸지 않는 단순 배경. 유전자 빈도와 무관하다."},
    "J": {"real": "개체 색상 표시 방식", "is_eq": False,
          "why": "AA·Aa·aa를 어떤 색으로 보여줄지일 뿐, 유전자 빈도와 무관하다."},
}
EQUILIBRIUM_LETTERS = [k for k, v in CONDITIONS.items() if v["is_eq"]]      # A,B,C,D,E
NEUTRAL_LETTERS     = [k for k, v in CONDITIONS.items() if not v["is_eq"]]  # F,G,H,I,J

# =====================================================================
# 2. 시뮬레이션 엔진
#    대립유전자 인코딩: 0 = A, 1 = a   /   유전자형 코드 = 두 대립유전자의 합
#    (0=AA, 1=Aa, 2=aa)
# =====================================================================
def _init_pop(N, p, rng):
    """초기 집단 생성: 각 대립유전자를 A(0)는 확률 p, a(1)는 확률 (1-p)로 뽑음."""
    return (rng.random((N, 2)) >= p).astype(np.int8)


def _step(alleles, prm, rng):
    """한 세대 진행: 선택 → 교배(+돌연변이) → 이주."""
    N = len(alleles)
    g = alleles.sum(axis=1)  # 0:AA, 1:Aa, 2:aa

    # --- 자연선택: 유전자형별 생존율을 가중치로 사용 ---
    surv = np.array([prm["s_AA"], prm["s_Aa"], prm["s_aa"]], dtype=float)
    w = surv[g]
    if w.sum() <= 0:
        w = np.ones(N)
    prob = w / w.sum()

    # --- 부모 1 선택 ---
    p1 = rng.choice(N, size=N, p=prob)

    # --- 부모 2 선택: 무작위 교배 vs 동형접합 선호(비무작위) ---
    if prm["mating"] == "random":
        p2 = rng.choice(N, size=N, p=prob)
    else:
        strength = prm.get("assort_strength", 0.85)  # 같은 유전자형끼리 교배할 확률
        p2 = rng.choice(N, size=N, p=prob)           # 기본은 무작위 짝
        assort_mask = rng.random(N) < strength
        gp1 = g[p1]
        for gt in (0, 1, 2):
            idx = np.where((gp1 == gt) & assort_mask)[0]
            if len(idx) == 0:
                continue
            pool = np.where(g == gt)[0]
            wp = prob[pool]
            wp = wp / wp.sum()
            p2[idx] = rng.choice(pool, size=len(idx), p=wp)

    # --- 각 부모가 대립유전자 하나씩 전달 ---
    a1 = alleles[p1, rng.integers(0, 2, size=N)]
    a2 = alleles[p2, rng.integers(0, 2, size=N)]
    new = np.stack([a1, a2], axis=1).astype(np.int8)

    # --- 돌연변이: 전달된 각 대립유전자가 확률 mu로 뒤집힘(A↔a) ---
    mu = prm["mu"]
    if mu > 0:
        flip = rng.random((N, 2)) < mu
        new = np.where(flip, 1 - new, new).astype(np.int8)

    # --- 이주: 일부 개체를 외부 유입 개체로 교체 ---
    m = prm["migration"]
    if m > 0:
        mask = rng.random(N) < m
        nm = int(mask.sum())
        if nm > 0:
            pa = prm["mig_A_freq"]
            mig = (rng.random((nm, 2)) >= pa).astype(np.int8)
            new[mask] = mig
    return new


@st.cache_data(show_spinner=False)
def run_simulation(N, generations, p0, mating, mu, migration,
                   mig_A_freq, s_AA, s_Aa, s_aa, assort_strength, seed):
    """전체 세대를 시뮬레이션하고 (DataFrame, 유전자형 이력)을 반환."""
    rng = np.random.default_rng(seed)
    prm = dict(N=N, p0=p0, mating=mating, mu=mu, migration=migration,
               mig_A_freq=mig_A_freq, s_AA=s_AA, s_Aa=s_Aa, s_aa=s_aa,
               assort_strength=assort_strength)

    alleles = _init_pop(N, p0, rng)
    rows = []
    geno_history = np.empty((generations + 1, N), dtype=np.int8)

    def record(al, gen):
        g = al.sum(axis=1)
        nAA = int((g == 0).sum()); nAa = int((g == 1).sum()); naa = int((g == 2).sum())
        p = (2 * nAA + nAa) / (2 * N)
        q = 1.0 - p
        rows.append(dict(
            세대=gen, p=p, q=q,
            관찰_AA=nAA / N, 관찰_Aa=nAa / N, 관찰_aa=naa / N,
            예상_AA=p * p, 예상_Aa=2 * p * q, 예상_aa=q * q,
        ))
        geno_history[gen] = g

    record(alleles, 0)
    for gen in range(1, generations + 1):
        alleles = _step(alleles, prm, rng)
        record(alleles, gen)

    return pd.DataFrame(rows), geno_history


# =====================================================================
# 3. 사이드바: 교사 모드 + 조건 A~J 조작
# =====================================================================
# 내부 매핑(학생에게는 숨김)
N_MAP   = {"수준 ①": 20, "수준 ②": 100, "수준 ③": 500, "수준 ④": 1000}
MU_MAP  = {"0%": 0.0, "1%": 0.01, "5%": 0.05}
MIG_MAP = {"없음": 0.0, "적음": 0.05, "많음": 0.20}

st.sidebar.title("🧪 실험실 조작판")
teacher = st.sidebar.toggle("👩‍🏫 교사용 모드 (조건의 진짜 의미 보기)", value=False)
seed = st.sidebar.number_input("🎲 난수 seed (같은 값=같은 결과)",
                               min_value=0, max_value=99999, value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("조건 A ~ J")


def label(letter):
    """교사 모드면 조건의 진짜 의미를 함께 표시."""
    base = f"조건 {letter}"
    if teacher:
        return f"{base}  ·  {CONDITIONS[letter]['real']}"
    return base


# --- 조건 A: 개체군 크기 (평형 O) ---
cA = st.sidebar.select_slider(label("A"), options=list(N_MAP.keys()), value="수준 ③")
# --- 조건 B: 무작위 교배 (평형 O) ---
cB = st.sidebar.radio(label("B"), ["방식 1", "방식 2"], horizontal=True, index=0,
                      help="짝을 어떻게 짓는지에 대한 두 가지 방식")
# --- 조건 C: 돌연변이 (평형 O) ---
cC = st.sidebar.select_slider(label("C"), options=list(MU_MAP.keys()), value="0%")
# --- 조건 D: 이주 (평형 O) ---
cD = st.sidebar.select_slider(label("D"), options=list(MIG_MAP.keys()), value="없음")
# --- 조건 E: 자연선택 (평형 O) ---
cE = st.sidebar.radio(label("E"), ["동일", "차이 있음"], horizontal=True, index=0,
                      help="세 유형의 생존율을 같게 / 다르게")

st.sidebar.markdown("· · ·")

# --- 조건 F: 초기 A 비율 (함정 X) ---
cF = st.sidebar.slider(label("F"), 0.1, 0.9, 0.5, 0.1)
# --- 조건 G: 관찰 세대 수 (함정 X) ---
cG = st.sidebar.slider(label("G"), 10, 200, 30, 5)
# --- 조건 H: 개체 이동 속도 (함정 X, 시각 전용) ---
cH = st.sidebar.select_slider(label("H"), options=["느림", "중간", "빠름"], value="중간")
# --- 조건 I: 배경 환경 (함정 X, 시각 전용) ---
cI = st.sidebar.selectbox(label("I"), ["흰색", "연회색", "연파랑", "연녹색"])
# --- 조건 J: 색상 표시 방식 (함정 X, 시각 전용) ---
cJ = st.sidebar.radio(label("J"), ["방식 1", "방식 2"], horizontal=True, index=0)

# 고급(교사용) 설정 — 이주민 A 비율
with st.sidebar.expander("⚙️ 고급 설정 (교사용)"):
    mig_A_freq = st.slider("이주민의 A 대립유전자 비율", 0.0, 1.0, 0.9, 0.1,
                           help="조건 D(이주)가 '적음/많음'일 때 들어오는 외부 개체의 A 비율")

# --- 조작값 → 시뮬레이션 파라미터로 변환 ---
N           = N_MAP[cA]
mating      = "random" if cB == "방식 1" else "assort"
mu          = MU_MAP[cC]
migration   = MIG_MAP[cD]
p0          = cF
generations = cG
if cE == "동일":
    s_AA, s_Aa, s_aa = 1.0, 1.0, 1.0
else:
    s_AA, s_Aa, s_aa = 1.0, 1.0, 0.7   # aa의 생존율이 낮음(약한 방향성 선택)
assort_strength = 0.85

# --- 시뮬레이션 실행 (캐시됨) ---
df, geno_history = run_simulation(
    N, generations, p0, mating, mu, migration,
    mig_A_freq, s_AA, s_Aa, s_aa, assort_strength, int(seed),
)

# 시각 전용 설정값
BG_MAP = {"흰색": "white", "연회색": "#f2f2f2", "연파랑": "#eef4fb", "연녹색": "#eef7ee"}
plot_bg = BG_MAP[cI]
if cJ == "방식 1":
    COLOR = {"AA": "#2563eb", "Aa": "#16a34a", "aa": "#dc2626"}  # 파/초/빨
else:
    COLOR = {"AA": "#7c3aed", "Aa": "#f59e0b", "aa": "#0891b2"}  # 보라/주황/청록
JITTER = {"느림": 0.15, "중간": 0.30, "빠름": 0.50}[cH]

# =====================================================================
# 4. 본문: 4개 화면(탭)
# =====================================================================
st.title("🧬 개체군은 언제 진화하지 않을까?")

tab1, tab2, tab3, tab4 = st.tabs(
    ["① 탐구 안내", "② 시뮬레이션", "③ 나의 판단", "④ 정답 공개·피드백"]
)

# ---------------------------------------------------------------------
# 화면 1. 탐구 안내
# ---------------------------------------------------------------------
with tab1:
    st.header("이 활동에서 할 일")
    st.markdown(
        """
        진화는 **대립유전자 빈도(p, q)의 변화**로 설명할 수 있습니다.
        그렇다면 거꾸로, **개체군이 진화하지 않으려면(=빈도가 변하지 않으려면)**
        어떤 조건이 필요할까요?

        왼쪽 조작판에는 **조건 A부터 J까지 10가지**가 있습니다.
        아직 각 조건이 무엇인지는 알려주지 않습니다.

        1. 조건을 **하나씩** 바꿔 보며 **②시뮬레이션** 탭에서 세대별 변화를 관찰하세요.
        2. **대립유전자 빈도(A, a)** 와 **유전자형 빈도(AA, Aa, aa)** 가
           세대가 지나도 **안정적으로 유지되는지** 살펴보세요.
        3. 활동 끝에 **③나의 판단** 탭에서 *평형 유지에 꼭 필요한 조건 5가지*를 고릅니다.
        4. 다 고르면 **④정답 공개** 탭에서 진짜 의미와 피드백을 확인합니다.
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.info(
            "**관찰 포인트 1 — 대립유전자 빈도**\n\n"
            "A와 a의 비율(p, q)이 세대가 지나도 그대로인가요, 변하나요?"
        )
    with c2:
        st.info(
            "**관찰 포인트 2 — 유전자형 빈도**\n\n"
            "AA : Aa : aa 비율이 하디-바인베르크 예상값 **p² : 2pq : q²** 와 일치하나요?"
        )
    st.warning(
        "💡 **실험 팁**: 한 번에 **한 조건만** 바꾸고 나머지는 기본값으로 두세요. "
        "그래야 그 조건이 빈도 변화의 원인인지 알 수 있습니다. "
        "또, 같은 조건이라도 **seed**(왼쪽 위)를 바꿔 여러 번 돌려 보면 "
        "*우연의 효과*가 큰 조건과 작은 조건을 구별할 수 있어요."
    )

# ---------------------------------------------------------------------
# 화면 2. 시뮬레이션
# ---------------------------------------------------------------------
with tab2:
    last = df.iloc[-1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("집단 크기", f"{N:,} 개체")
    m2.metric("관찰 세대", f"{generations} 세대")
    m3.metric("A 빈도 p (처음→끝)", f"{df.iloc[0]['p']:.2f} → {last['p']:.2f}",
              delta=f"{last['p'] - df.iloc[0]['p']:+.2f}")
    m4.metric("a 빈도 q (처음→끝)", f"{df.iloc[0]['q']:.2f} → {last['q']:.2f}",
              delta=f"{last['q'] - df.iloc[0]['q']:+.2f}")

    st.markdown("### 🐢 개체군 모습")
    show_gen = st.slider("표시할 세대", 0, generations, generations, key="popgen")

    left, right = st.columns([1, 1.3])

    # --- (좌) 개체군 산점도 ---
    with left:
        g_disp = geno_history[show_gen]
        rng_pos = np.random.default_rng(1000 + show_gen)  # 세대별 고정 위치
        n_disp = len(g_disp)
        side = int(np.ceil(np.sqrt(n_disp)))
        gx, gy = np.meshgrid(np.arange(side), np.arange(side))
        gx = gx.flatten()[:n_disp].astype(float)
        gy = gy.flatten()[:n_disp].astype(float)
        gx += rng_pos.uniform(-JITTER, JITTER, n_disp)  # 이동 속도(H) = 흔들림(시각 전용)
        gy += rng_pos.uniform(-JITTER, JITTER, n_disp)

        names = np.array(["AA", "Aa", "aa"])[g_disp]
        fig_pop = go.Figure()
        for gt in ["AA", "Aa", "aa"]:
            sel = names == gt
            fig_pop.add_trace(go.Scatter(
                x=gx[sel], y=gy[sel], mode="markers", name=gt,
                marker=dict(size=max(4, 220 / side), color=COLOR[gt],
                            line=dict(width=0)),
            ))
        fig_pop.update_layout(
            height=380, plot_bgcolor=plot_bg, paper_bgcolor=plot_bg,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
            legend=dict(orientation="h", y=1.08),
            title=f"{show_gen}세대 ({n_disp:,}개체)",
        )
        st.plotly_chart(fig_pop, width='stretch')

    # --- (우) 현재 세대 표: 관찰 vs 예상 ---
    with right:
        row = df.iloc[show_gen]
        st.markdown(f"#### 📋 {show_gen}세대 빈도표")
        tbl = pd.DataFrame({
            "유전자형": ["AA", "Aa", "aa"],
            "관찰 빈도": [row["관찰_AA"], row["관찰_Aa"], row["관찰_aa"]],
            "예상 빈도 (p²,2pq,q²)": [row["예상_AA"], row["예상_Aa"], row["예상_aa"]],
        })
        tbl["관찰 − 예상"] = tbl["관찰 빈도"] - tbl["예상 빈도 (p²,2pq,q²)"]
        st.dataframe(
            tbl.style.format({
                "관찰 빈도": "{:.3f}", "예상 빈도 (p²,2pq,q²)": "{:.3f}",
                "관찰 − 예상": "{:+.3f}",
            }),
            hide_index=True, width='stretch',
        )
        st.caption(
            f"이 세대의 p = {row['p']:.3f}, q = {row['q']:.3f}  ·  "
            "관찰값과 예상값이 크게 다르면 **무작위 교배가 아닐 가능성**이 큽니다."
        )

    st.markdown("### 📈 세대별 변화 그래프")
    gcol1, gcol2 = st.columns(2)

    # --- 대립유전자 빈도 그래프 ---
    with gcol1:
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=df["세대"], y=df["p"], name="A (p)",
                                   line=dict(color="#1d4ed8", width=3)))
        fig_a.add_trace(go.Scatter(x=df["세대"], y=df["q"], name="a (q)",
                                   line=dict(color="#b91c1c", width=3)))
        fig_a.update_layout(
            title="대립유전자 빈도", height=340, plot_bgcolor=plot_bg,
            paper_bgcolor=plot_bg, yaxis=dict(range=[0, 1], title="빈도"),
            xaxis=dict(title="세대"), legend=dict(orientation="h", y=1.12),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_a, width='stretch')

    # --- 유전자형 빈도 그래프 (관찰 실선 + 예상 점선) ---
    with gcol2:
        fig_g = go.Figure()
        for gt, col in [("AA", "관찰_AA"), ("Aa", "관찰_Aa"), ("aa", "관찰_aa")]:
            fig_g.add_trace(go.Scatter(
                x=df["세대"], y=df[col], name=f"{gt} 관찰",
                line=dict(color=COLOR[gt], width=3)))
        for gt, col in [("AA", "예상_AA"), ("Aa", "예상_Aa"), ("aa", "예상_aa")]:
            fig_g.add_trace(go.Scatter(
                x=df["세대"], y=df[col], name=f"{gt} 예상",
                line=dict(color=COLOR[gt], width=1.5, dash="dot"),
                opacity=0.6))
        fig_g.update_layout(
            title="유전자형 빈도 (실선=관찰, 점선=예상 p²·2pq·q²)",
            height=340, plot_bgcolor=plot_bg, paper_bgcolor=plot_bg,
            yaxis=dict(range=[0, 1], title="빈도"), xaxis=dict(title="세대"),
            legend=dict(orientation="h", y=1.18, font=dict(size=10)),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_g, width='stretch')

    with st.expander("📊 세대별 원자료(표) 보기 / 내려받기"):
        show_df = df.copy()
        for c in show_df.columns[1:]:
            show_df[c] = show_df[c].round(4)
        st.dataframe(show_df, hide_index=True, width='stretch')
        st.download_button("CSV 내려받기", show_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="simulation_data.csv", mime="text/csv")

# ---------------------------------------------------------------------
# 화면 3. 나의 판단
# ---------------------------------------------------------------------
with tab3:
    st.header("내가 찾은 평형 조건")
    st.markdown(
        "여러 조건을 충분히 실험해 봤다면, 아래에 결론을 정리하세요. "
        "**평형 유지에 꼭 필요한 조건 5개**와 **관련 없다고 생각하는 조건 5개**를 고릅니다."
    )

    all_letters = list(CONDITIONS.keys())
    picks = st.multiselect(
        "✅ 평형 유지에 **필요하다**고 생각하는 조건 (정확히 5개)",
        options=all_letters,
        default=st.session_state.get("picks", []),
        key="picks",
    )
    chosen = set(picks)
    not_needed = [x for x in all_letters if x not in chosen]
    st.caption(f"선택: {len(picks)}개  ·  자동으로 '관련 없음'으로 분류됨: "
               f"{', '.join(not_needed) if not_needed else '—'}")

    st.markdown("#### 관찰 기록 (탐구 질문)")
    q1 = st.text_area("1) 조건을 바꾸었을 때 **대립유전자 빈도(p, q)** 가 변한 조건은? 어떻게 변했나요?",
                      key="q1", height=80)
    q2 = st.text_area("2) 조건을 바꾸었을 때 **유전자형 빈도(AA·Aa·aa)** 가 예상값에서 벗어난 조건은?",
                      key="q2", height=80)
    q3 = st.text_area("3) 그 변화가 **여러 세대에 걸쳐 유지**되었나요? (한 번 출렁였다 돌아오는지 vs 계속 변하는지)",
                      key="q3", height=80)
    q4 = st.text_area("4) 내가 고른 5개를 **평형 조건이라고 판단한 근거**(그래프 관찰)는 무엇인가요?",
                      key="q4", height=100)

    if len(picks) != 5:
        st.error(f"평형 조건은 **정확히 5개**를 골라야 합니다. (현재 {len(picks)}개)")
    else:
        st.success("5개 선택 완료! **④정답 공개** 탭에서 결과를 확인하세요.")

# ---------------------------------------------------------------------
# 화면 4. 정답 공개·피드백
# ---------------------------------------------------------------------
with tab4:
    st.header("정답 공개 및 피드백")

    picks = st.session_state.get("picks", [])
    reveal = teacher or st.button("🔓 정답 공개하기", type="primary")

    if not reveal:
        st.info("**③나의 판단**에서 5개를 고른 뒤, 위 버튼을 누르거나 "
                "교사용 모드를 켜면 정답이 공개됩니다.")
    else:
        chosen = set(picks)
        rows = []
        for L, info in CONDITIONS.items():
            picked = L in chosen
            correct = (picked == info["is_eq"])
            rows.append({
                "조건": L,
                "실제 의미": info["real"],
                "평형 조건?": "O" if info["is_eq"] else "X",
                "내 선택": "필요" if picked else "관련 없음",
                "채점": "✅" if correct else "❌",
            })
        result = pd.DataFrame(rows)
        st.dataframe(result, hide_index=True, width='stretch')

        if picks:
            n_correct = int((result["채점"] == "✅").sum())
            st.metric("정답 수", f"{n_correct} / 10")
            hit = sorted(chosen & set(EQUILIBRIUM_LETTERS))
            miss = sorted(set(EQUILIBRIUM_LETTERS) - chosen)
            trap = sorted(chosen & set(NEUTRAL_LETTERS))
            if hit:
                st.success("바르게 찾은 평형 조건: " +
                           ", ".join(f"{L}({CONDITIONS[L]['real']})" for L in hit))
            if miss:
                st.warning("놓친 평형 조건: " +
                           ", ".join(f"{L}({CONDITIONS[L]['real']})" for L in miss))
            if trap:
                st.error("함정에 걸린 조건(평형 조건 아님): " +
                         ", ".join(f"{L}({CONDITIONS[L]['real']})" for L in trap))

        st.markdown("### 🔑 하디-바인베르크 평형 5조건")
        for L in EQUILIBRIUM_LETTERS:
            st.markdown(f"- **{L} · {CONDITIONS[L]['real']}** — {CONDITIONS[L]['why']}")

        st.markdown("### ⚠️ 자주 하는 오개념 바로잡기")
        st.markdown(
            f"""
- **초기 A 비율(F)** 은 시작값과 p²·2pq·q² *값*을 바꾸지만, 평형이 **유지되는지**와는 무관합니다.
  → 평형 조건을 다 만족하면 p가 0.2든 0.8든 그 값에서 그대로 유지됩니다.
- **개체군이 작으면(A)** 자연선택이 없어도 **우연(유전적 부동)** 만으로 빈도가 출렁이고,
  심하면 한 대립유전자가 사라질 수 있습니다. seed를 바꿔 가며 확인해 보세요.
- **자연선택(E)** 은 유전자형의 *생존율 차이*를 통해 특정 대립유전자를 늘리거나 줄입니다.
- **돌연변이(C)·이주(D)** 는 대립유전자를 바꾸거나 외부에서 들여와 빈도를 변화시킵니다.
- **무작위 교배(B)** 가 깨지면 **대립유전자 빈도는 거의 그대로**여도
  유전자형 빈도가 **p²:2pq:q²에서 벗어납니다**(동형접합 과잉, 이형접합 부족).
  → ②탭에서 '관찰 − 예상' 차이가 가장 잘 드러나는 조건입니다.
- **세대 수(G)·이동 속도(H)·배경(I)·색상(J)** 은 관찰/표시 방식일 뿐, 유전자 빈도와 무관합니다.
            """
        )

# 푸터
st.sidebar.markdown("---")
st.sidebar.caption("고등학교 생명과학 · 하디-바인베르크 평형 탐구 · Streamlit")
