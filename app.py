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
# 1. 조건 A~J 메타데이터 (내부 정답 — 학생에겐 숨김)
# =====================================================================
CONDITIONS = {
    "A": {"real": "개체군 크기", "is_eq": True,
          "why": "집단이 작으면 우연(유전적 부동)만으로도 유전자 비율이 크게 흔들릴 수 있어요."},
    "B": {"real": "무작위 짝짓기", "is_eq": True,
          "why": "짝짓기가 무작위가 아니면, 유전자 비율은 거의 그대로여도 유전자형 비율이 예상값에서 벗어나요."},
    "C": {"real": "돌연변이 없음", "is_eq": True,
          "why": "돌연변이는 한 유전자를 다른 유전자로 바꾸어 비율을 변하게 해요."},
    "D": {"real": "이주(외부 유입) 없음", "is_eq": True,
          "why": "바깥에서 개체가 들어오면 다른 유전자 비율이 섞여 비율이 변해요."},
    "E": {"real": "자연선택 없음", "is_eq": True,
          "why": "유형마다 생존율이 다르면 특정 유전자가 점점 늘거나 줄어들어요."},
    "F": {"real": "처음 A 유전자 비율", "is_eq": False,
          "why": "시작값과 예상값(p²·2pq·q²)은 달라지지만, 평형이 '유지되는지'와는 상관없어요."},
    "G": {"real": "관찰 세대 수", "is_eq": False,
          "why": "오래 본다고 평형이 깨지거나 만들어지지 않아요. 그냥 관찰 기간일 뿐."},
    "H": {"real": "개체 움직이는 속도", "is_eq": False,
          "why": "화면에서 점이 흔들리는 정도일 뿐, 유전자 비율과는 관계없어요."},
    "I": {"real": "배경 색(서식지)", "is_eq": False,
          "why": "생존율을 바꾸지 않는 단순 배경. 유전자 비율과 무관해요."},
    "J": {"real": "개체 색칠 방식", "is_eq": False,
          "why": "유형을 어떤 색으로 보여줄지일 뿐, 유전자 비율과 무관해요."},
}
EQ_LETTERS = [k for k, v in CONDITIONS.items() if v["is_eq"]]       # A,B,C,D,E
NEUTRAL_LETTERS = [k for k, v in CONDITIONS.items() if not v["is_eq"]]  # F,G,H,I,J

# ===== 조건 힌트(수수께끼 + 애니메이션) =====
HINT = {
    "A": "한산한 무리 🐾 vs 북적이는 무리. 수가 적으면 ‘우연’이 더 크게 작용할까?",
    "B": "💞 누구와 짝을 이룰까? 아무나? 아니면 닮은 끼리끼리?",
    "C": "✨ 가끔 글자가 슬쩍 바뀐다. A가 a로, a가 A로?",
    "D": "🚪 담장 너머에서 새 친구가 들어온다.",
    "E": "🌿 누구는 더 잘 살아남는다. 모두 똑같이 살아남을까?",
    "F": "🏁 출발선에서 파랑·빨강 비율을 정한다. 그냥 ‘시작값’일 뿐일까?",
    "G": "⏳ 얼마나 오래(몇 세대) 지켜볼까? 오래 본다고 달라질까?",
    "H": "🏃 점이 빠르게 돌아다닌다. 빨리 움직이면 유전자도 변할까?",
    "I": "🏞️ 배경 풍경 색을 바꾼다. 색이 생존을 바꿀까?",
    "J": "🎨 같은 친구를 다른 색으로 칠한다. 색만 바뀌면 비율도 바뀔까?",
}

def _svg(body, vb="0 0 200 96"):
    return (f'<div style="display:flex;justify-content:center;">'
            f'<svg viewBox="{vb}" width="200" xmlns="http://www.w3.org/2000/svg">{body}</svg></div>')

ANIM = {}
# A: 적은 무리 vs 많은 무리
ANIM["A"] = _svg('''
<text x="42" y="12" font-size="10" text-anchor="middle" fill="#666">적은 무리</text>
<text x="155" y="12" font-size="10" text-anchor="middle" fill="#666">많은 무리</text>
<line x1="100" y1="20" x2="100" y2="92" stroke="#ddd" stroke-dasharray="3 3"/>
<circle cx="30" cy="45" r="6" fill="#2563eb"/><circle cx="52" cy="60" r="6" fill="#dc2626"/>
<circle cx="40" cy="78" r="6" fill="#2563eb"/>
''' + ''.join(
    f'<circle cx="{120+(i%4)*20}" cy="{35+(i//4)*18}" r="5" fill="{["#2563eb","#dc2626","#16a34a"][i%3]}">'
    f'<animate attributeName="opacity" values="1;0.3;1" dur="1.6s" begin="{i*0.1}s" repeatCount="indefinite"/></circle>'
    for i in range(12)))

# B: 끼리끼리 만나는 두 점 + 하트
ANIM["B"] = _svg('''
<circle cx="35" cy="50" r="9" fill="#2563eb"><animate attributeName="cx" values="35;78;35" dur="2.4s" repeatCount="indefinite"/></circle>
<circle cx="165" cy="50" r="9" fill="#2563eb"><animate attributeName="cx" values="165;122;165" dur="2.4s" repeatCount="indefinite"/></circle>
<text x="100" y="56" font-size="20" text-anchor="middle">💞
<animate attributeName="opacity" values="0;0;1;0" dur="2.4s" repeatCount="indefinite"/></text>
<text x="100" y="86" font-size="9" text-anchor="middle" fill="#888">끼리끼리?</text>
''')

# C: A가 a로 바뀌는 돌연변이
ANIM["C"] = _svg('''
<text x="100" y="60" font-size="40" text-anchor="middle" fill="#2563eb" font-weight="bold">A
<animate attributeName="opacity" values="1;1;0;0;1" dur="3s" repeatCount="indefinite"/></text>
<text x="100" y="60" font-size="40" text-anchor="middle" fill="#dc2626" font-weight="bold">a
<animate attributeName="opacity" values="0;0;1;1;0" dur="3s" repeatCount="indefinite"/></text>
<text x="140" y="30" font-size="22" text-anchor="middle">✨
<animate attributeName="opacity" values="0;1;0" dur="3s" begin="1.2s" repeatCount="indefinite"/></text>
''')

# D: 바깥에서 들어오는 이주
ANIM["D"] = _svg('''
<rect x="70" y="22" width="110" height="60" rx="8" fill="none" stroke="#999" stroke-width="2"/>
<circle cx="110" cy="55" r="7" fill="#2563eb"/><circle cx="150" cy="45" r="7" fill="#2563eb"/>
<circle cx="135" cy="70" r="7" fill="#dc2626"/>
<circle cx="10" cy="52" r="7" fill="#16a34a"><animate attributeName="cx" values="10;95;95" dur="2.6s" repeatCount="indefinite"/>
<animate attributeName="opacity" values="1;1;0" dur="2.6s" repeatCount="indefinite"/></circle>
<text x="30" y="86" font-size="9" text-anchor="middle" fill="#888">바깥 친구</text>
''')

# E: 일부만 살아남는 선택
ANIM["E"] = _svg('''
''' + ''.join(
    f'<circle cx="{30+i*35}" cy="48" r="9" fill="{["#2563eb","#dc2626","#2563eb","#dc2626","#2563eb"][i]}">'
    + ('<animate attributeName="opacity" values="1;1;0.1;0.1;1" dur="3s" repeatCount="indefinite"/>' if i in (1,3) else '')
    + '</circle>'
    + (f'<text x="{30+i*35}" y="53" font-size="14" text-anchor="middle" fill="#fff">✕<animate attributeName="opacity" values="0;0;1;1;0" dur="3s" repeatCount="indefinite"/></text>' if i in (1,3) else '')
    for i in range(5))
+ '<text x="100" y="84" font-size="9" text-anchor="middle" fill="#888">누가 살아남나?</text>')

# F: 출발선 비율 막대
ANIM["F"] = _svg('''
<text x="100" y="14" font-size="10" text-anchor="middle" fill="#666">🏁 출발 비율</text>
<rect x="30" y="35" width="140" height="26" rx="4" fill="#dc2626"/>
<rect x="30" y="35" width="80" height="26" rx="4" fill="#2563eb">
<animate attributeName="width" values="80;110;50;80" dur="4s" repeatCount="indefinite"/></rect>
<text x="100" y="82" font-size="9" text-anchor="middle" fill="#888">시작값일 뿐?</text>
''')

# G: 세대 카운터(모래시계 느낌)
ANIM["G"] = _svg('''
<text x="100" y="58" font-size="34" text-anchor="middle">⏳</text>
<text x="150" y="40" font-size="16" text-anchor="middle" fill="#2563eb" font-weight="bold">1
<animate attributeName="opacity" values="1;0" dur="0.9s" repeatCount="indefinite"/></text>
<text x="150" y="40" font-size="16" text-anchor="middle" fill="#16a34a" font-weight="bold">2
<animate attributeName="opacity" values="0;1;0" dur="0.9s" repeatCount="indefinite"/></text>
<text x="100" y="86" font-size="9" text-anchor="middle" fill="#888">몇 세대 볼까?</text>
''')

# H: 빠르게 지나가는 점
ANIM["H"] = _svg('''
<circle cx="20" cy="50" r="8" fill="#16a34a"><animate attributeName="cx" values="20;180;20" dur="1s" repeatCount="indefinite"/></circle>
<text x="100" y="84" font-size="9" text-anchor="middle" fill="#888">🏃 빨리 움직이면?</text>
''')

# I: 배경색이 바뀜
ANIM["I"] = _svg('''
<rect x="25" y="25" width="150" height="45" rx="8">
<animate attributeName="fill" values="#eef4fb;#eef7ee;#f7f0ee;#eef4fb" dur="3s" repeatCount="indefinite"/></rect>
<circle cx="100" cy="47" r="9" fill="#2563eb"/>
<text x="100" y="86" font-size="9" text-anchor="middle" fill="#888">🏞️ 배경만 바뀜</text>
''')

# J: 같은 점의 색만 순환
ANIM["J"] = _svg('''
<circle cx="100" cy="46" r="16">
<animate attributeName="fill" values="#2563eb;#16a34a;#dc2626;#7c3aed;#2563eb" dur="2.4s" repeatCount="indefinite"/></circle>
<text x="100" y="84" font-size="9" text-anchor="middle" fill="#888">🎨 색만 바뀜</text>
''')

# =====================================================================
# 2. 시뮬레이션 엔진  (대립유전자 0=A, 1=a / 유전자형 = 두 값의 합)
# =====================================================================
def _init_pop(N, p, rng):
    return (rng.random((N, 2)) >= p).astype(np.int8)

def _step(alleles, prm, rng):
    N = len(alleles)
    g = alleles.sum(axis=1)  # 0:AA, 1:Aa, 2:aa
    surv = np.array([prm["s_AA"], prm["s_Aa"], prm["s_aa"]], dtype=float)
    w = surv[g]
    if w.sum() <= 0:
        w = np.ones(N)
    prob = w / w.sum()

    p1 = rng.choice(N, size=N, p=prob)
    if prm["mating"] == "random":
        p2 = rng.choice(N, size=N, p=prob)
    else:
        strength = prm.get("assort_strength", 0.85)
        p2 = rng.choice(N, size=N, p=prob)
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

    a1 = alleles[p1, rng.integers(0, 2, size=N)]
    a2 = alleles[p2, rng.integers(0, 2, size=N)]
    new = np.stack([a1, a2], axis=1).astype(np.int8)

    mu = prm["mu"]
    if mu > 0:
        flip = rng.random((N, 2)) < mu
        new = np.where(flip, 1 - new, new).astype(np.int8)

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
        rows.append(dict(세대=gen, p=p, q=q,
                         관찰_AA=nAA / N, 관찰_Aa=nAa / N, 관찰_aa=naa / N,
                         예상_AA=p * p, 예상_Aa=2 * p * q, 예상_aa=q * q))
        geno_history[gen] = g

    record(alleles, 0)
    for gen in range(1, generations + 1):
        alleles = _step(alleles, prm, rng)
        record(alleles, gen)
    return pd.DataFrame(rows), geno_history

# =====================================================================
# 3. 기본값 + 되돌리기(리셋) 처리
# =====================================================================
DEFAULTS = {
    "kSeed": 42, "kA": "보통 (③)", "kB": "방식 1", "kC": "0%", "kD": "없음",
    "kE": "똑같음", "kF": 0.5, "kG": 30, "kH": "보통", "kI": "흰색",
    "kJ": "방식 1", "kMig": 0.9,
}
# 첫 실행 시 기본값 세팅
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
# 되돌리기 플래그 처리
if st.session_state.get("_reset"):
    for _k, _v in DEFAULTS.items():
        st.session_state[_k] = _v
    st.session_state["_reset"] = False

# 내부 매핑
N_MAP = {"아주 작음 (①)": 20, "작음 (②)": 100, "보통 (③)": 500, "큼 (④)": 1000}
MU_MAP = {"0%": 0.0, "1%": 0.01, "5%": 0.05}
MIG_MAP = {"없음": 0.0, "적음": 0.05, "많음": 0.20}

# =====================================================================
# 4. 사이드바 : 조작 패널
# =====================================================================
st.sidebar.title("🎛️ 조작 패널")
st.sidebar.caption("값을 바꾸면 결과가 **바로** 바뀌어요.")

c_left, c_right = st.sidebar.columns([1, 1])
if c_left.button("↩️ 처음 상태로", width='stretch',
                 help="모든 조건을 기본값으로 되돌려요."):
    st.session_state["_reset"] = True
    st.rerun()
teacher = c_right.toggle("👩‍🏫 선생님", value=False,
                         help="켜면 조건 A~J의 진짜 의미가 보여요.")

show_hint = st.sidebar.toggle(
    "💡 조건 힌트 켜기", value=False,
    help="켜면 각 조건 아래에 ‘무엇을 바꾸는지’ 살짝 힌트가 나와요. (정답 아님)")

st.sidebar.number_input(
    "🎲 실험 번호", min_value=0, max_value=99999, step=1, key="kSeed",
    help="같은 번호로 돌리면 결과가 똑같이 나와요. 번호를 바꾸면 '우연'이 어떻게 작용하는지 볼 수 있어요.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("조건 A ~ J")
st.sidebar.caption("💡 **한 번에 하나씩만** 바꿔 보고, 그래프가 변하는지 관찰하세요.")

def label(letter):
    base = f"조건 {letter}"
    return f"{base}  ·  {CONDITIONS[letter]['real']}" if teacher else base

st.sidebar.select_slider(label("A"), options=list(N_MAP.keys()), key="kA")
if show_hint:
    st.sidebar.caption(HINT["A"])
st.sidebar.radio(label("B"), ["방식 1", "방식 2"], horizontal=True, key="kB")
if show_hint:
    st.sidebar.caption(HINT["B"])
st.sidebar.select_slider(label("C"), options=list(MU_MAP.keys()), key="kC")
if show_hint:
    st.sidebar.caption(HINT["C"])
st.sidebar.select_slider(label("D"), options=list(MIG_MAP.keys()), key="kD")
if show_hint:
    st.sidebar.caption(HINT["D"])
st.sidebar.radio(label("E"), ["똑같음", "다름"], horizontal=True, key="kE")
if show_hint:
    st.sidebar.caption(HINT["E"])
st.sidebar.markdown("· · ·")
st.sidebar.slider(label("F"), 0.1, 0.9, step=0.1, key="kF")
if show_hint:
    st.sidebar.caption(HINT["F"])
st.sidebar.slider(label("G"), 10, 200, step=5, key="kG")
if show_hint:
    st.sidebar.caption(HINT["G"])
st.sidebar.select_slider(label("H"), options=["느림", "보통", "빠름"], key="kH")
if show_hint:
    st.sidebar.caption(HINT["H"])
st.sidebar.selectbox(label("I"), ["흰색", "연회색", "연파랑", "연녹색"], key="kI")
if show_hint:
    st.sidebar.caption(HINT["I"])
st.sidebar.radio(label("J"), ["방식 1", "방식 2"], horizontal=True, key="kJ")
if show_hint:
    st.sidebar.caption(HINT["J"])

with st.sidebar.expander("⚙️ 고급 설정 (선생님용)"):
    st.slider("이주민의 A 유전자 비율", 0.0, 1.0, step=0.1, key="kMig",
              help="조건 D(이주)가 '적음/많음'일 때 들어오는 외부 개체의 A 비율")

with st.sidebar.expander("❓ 사용법 도움말"):
    st.markdown(
        "- 왼쪽 조건을 바꾸면 오른쪽 그래프가 바로 바뀌어요.\n"
        "- **‘② 실험하기’** 탭에서 세대별 변화를 관찰하세요.\n"
        "- 헷갈리면 **‘↩️ 처음 상태로’** 를 눌러 다시 시작하세요.\n"
        "- 같은 조건이라도 **실험 번호**를 바꿔 여러 번 돌려 보세요."
    )

# --- 조작값 → 파라미터 ---
N = N_MAP[st.session_state["kA"]]
mating = "random" if st.session_state["kB"] == "방식 1" else "assort"
mu = MU_MAP[st.session_state["kC"]]
migration = MIG_MAP[st.session_state["kD"]]
p0 = st.session_state["kF"]
generations = st.session_state["kG"]
if st.session_state["kE"] == "똑같음":
    s_AA, s_Aa, s_aa = 1.0, 1.0, 1.0
else:
    s_AA, s_Aa, s_aa = 1.0, 1.0, 0.7
mig_A_freq = st.session_state["kMig"]
seed = int(st.session_state["kSeed"])

df, geno_history = run_simulation(
    N, generations, p0, mating, mu, migration,
    mig_A_freq, s_AA, s_Aa, s_aa, 0.85, seed,
)

# 시각 전용 설정
BG_MAP = {"흰색": "white", "연회색": "#f2f2f2", "연파랑": "#eef4fb", "연녹색": "#eef7ee"}
plot_bg = BG_MAP[st.session_state["kI"]]
if st.session_state["kJ"] == "방식 1":
    COLOR = {"AA": "#2563eb", "Aa": "#16a34a", "aa": "#dc2626"}
else:
    COLOR = {"AA": "#7c3aed", "Aa": "#f59e0b", "aa": "#0891b2"}
JITTER = {"느림": 0.15, "보통": 0.30, "빠름": 0.50}[st.session_state["kH"]]

# =====================================================================
# 5. 본문 : 탭 4개
# =====================================================================
st.title("🧬 개체군은 언제 진화하지 않을까?")

tab1, tabH, tab2, tab3, tab4 = st.tabs(
    ["① 시작하기", "💡 조건 힌트", "② 실험하기", "③ 내 생각 정리", "④ 결과 확인"]
)

# ---------------------------------------------------------------------
# 화면 1. 시작하기
# ---------------------------------------------------------------------
with tab1:
    st.header("무엇을 하는 활동일까요?")
    st.markdown(
        """
        생물의 **진화**는 한마디로 **유전자 비율이 바뀌는 것**이에요.
        그렇다면 거꾸로, **개체군이 진화하지 *않으려면*(=유전자 비율이 안 바뀌려면)**
        어떤 조건이 필요할까요? 그걸 **직접 실험해서 찾아내는** 활동입니다.
        """
    )
    st.markdown("#### 이렇게 해보세요 🔍")
    st.markdown(
        """
        1. 왼쪽 **조작 패널**에 조건 **A~J**가 있어요. (아직 무엇인지는 비밀!)
        2. **‘② 실험하기’** 탭에서 조건을 **하나씩** 바꿔 보며 변화를 관찰해요.
        3. **‘③ 내 생각 정리’** 탭에서 *평형에 꼭 필요한 조건 5개*를 골라요.
        4. **‘④ 결과 확인’** 탭에서 진짜 정답과 설명을 확인해요!
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.info(
            "**관찰 포인트 1 · 유전자 비율**\n\n"
            "A와 a의 비율이 세대가 지나도 **그대로**인가요, 아니면 **변하나요**?"
        )
    with c2:
        st.info(
            "**관찰 포인트 2 · 유전자형 비율**\n\n"
            "AA : Aa : aa 비율이 **예상값(p² : 2pq : q²)** 과 **일치**하나요?"
        )
    st.success(
        "💡 **실험 비법**\n\n"
        "• 한 번에 **하나의 조건만** 바꾸세요. 그래야 원인을 알 수 있어요.\n\n"
        "• 길을 잃으면 왼쪽 **‘↩️ 처음 상태로’** 버튼을 누르세요.\n\n"
        "• 같은 조건이라도 **실험 번호**를 바꿔 여러 번 돌려 보면 *우연의 힘*이 큰 조건을 찾을 수 있어요."
    )

# ---------------------------------------------------------------------
# 화면 H. 조건 힌트 (애니메이션 + 수수께끼)
# ---------------------------------------------------------------------
with tabH:
    st.header("💡 조건이 ‘무엇을 바꾸는지’ 힌트")
    st.markdown(
        "각 조건이 **어떤 것을 건드리는지** 살짝 보여주는 힌트예요. "
        "**정답(평형 조건인지 아닌지)** 은 직접 **‘② 실험하기’** 에서 확인하세요! 🔍"
    )
    letters = list(CONDITIONS.keys())
    for i in range(0, len(letters), 2):
        cols = st.columns(2)
        for j, L in enumerate(letters[i:i + 2]):
            with cols[j]:
                title = f"조건 {L}"
                if teacher:
                    title += f"  ·  {CONDITIONS[L]['real']}"
                st.markdown(f"**{title}**")
                st.html(ANIM[L])
                st.caption(HINT[L])

# ---------------------------------------------------------------------
# 화면 2. 실험하기
# ---------------------------------------------------------------------
with tab2:
    first, last = df.iloc[0], df.iloc[-1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("전체 개체 수", f"{N:,} 마리")
    m2.metric("관찰한 세대", f"{generations} 세대")
    m3.metric("A 유전자 비율 (시작 → 지금)", f"{first['p']:.2f} → {last['p']:.2f}",
              delta=f"{last['p'] - first['p']:+.2f}")
    m4.metric("a 유전자 비율 (시작 → 지금)", f"{first['q']:.2f} → {last['q']:.2f}",
              delta=f"{last['q'] - first['q']:+.2f}")
    st.caption("위 숫자의 변화량(▲▼)이 0에 가까울수록 '진화가 일어나지 않은' 거예요.")

    st.markdown("### 🐢 개체군 들여다보기")
    show_gen = st.slider("몇 세대째를 볼까요?", 0, generations, generations, key="popgen")

    left, right = st.columns([1, 1.3])
    with left:
        g_disp = geno_history[show_gen]
        rng_pos = np.random.default_rng(1000 + show_gen)
        n_disp = len(g_disp)
        side = int(np.ceil(np.sqrt(n_disp)))
        gx, gy = np.meshgrid(np.arange(side), np.arange(side))
        gx = gx.flatten()[:n_disp].astype(float)
        gy = gy.flatten()[:n_disp].astype(float)
        gx += rng_pos.uniform(-JITTER, JITTER, n_disp)
        gy += rng_pos.uniform(-JITTER, JITTER, n_disp)
        names = np.array(["AA", "Aa", "aa"])[g_disp]
        fig_pop = go.Figure()
        for gt in ["AA", "Aa", "aa"]:
            sel = names == gt
            fig_pop.add_trace(go.Scatter(
                x=gx[sel], y=gy[sel], mode="markers", name=gt,
                marker=dict(size=max(4, 220 / side), color=COLOR[gt], line=dict(width=0))))
        fig_pop.update_layout(
            height=380, plot_bgcolor=plot_bg, paper_bgcolor=plot_bg,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
            legend=dict(orientation="h", y=1.08),
            title=f"{show_gen}세대 ({n_disp:,}마리)")
        st.plotly_chart(fig_pop, width='stretch')
        st.caption("점 하나가 개체 한 마리예요. AA · Aa · aa는 한 유전자의 세 가지 유형이에요.")

    with right:
        row = df.iloc[show_gen]
        st.markdown(f"#### 📋 {show_gen}세대 비율표")
        tbl = pd.DataFrame({
            "유전자형": ["AA", "Aa", "aa"],
            "실제 관찰한 비율": [row["관찰_AA"], row["관찰_Aa"], row["관찰_aa"]],
            "이론이 예상한 비율": [row["예상_AA"], row["예상_Aa"], row["예상_aa"]],
        })
        tbl["차이 (관찰−예상)"] = tbl["실제 관찰한 비율"] - tbl["이론이 예상한 비율"]
        st.dataframe(
            tbl.style.format({
                "실제 관찰한 비율": "{:.3f}", "이론이 예상한 비율": "{:.3f}",
                "차이 (관찰−예상)": "{:+.3f}"}),
            hide_index=True, width='stretch')
        st.caption(
            f"이 세대의 A 비율 p = {row['p']:.3f}, a 비율 q = {row['q']:.3f}\n\n"
            "👀 **차이가 큰지 작은지** 잘 살펴보세요. 어떤 조건에서 차이가 커지나요?"
        )

    st.markdown("### 📈 세대별 변화 그래프")
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=df["세대"], y=df["p"], name="A 유전자 (p)",
                                   line=dict(color="#1d4ed8", width=3)))
        fig_a.add_trace(go.Scatter(x=df["세대"], y=df["q"], name="a 유전자 (q)",
                                   line=dict(color="#b91c1c", width=3)))
        fig_a.update_layout(
            title="① 유전자 비율 변화", height=340, plot_bgcolor=plot_bg,
            paper_bgcolor=plot_bg, yaxis=dict(range=[0, 1], title="비율"),
            xaxis=dict(title="세대"), legend=dict(orientation="h", y=1.12),
            margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_a, width='stretch')
        st.caption("선이 평평하면 → 유전자 비율이 안 변하는(=진화가 없는) 거예요.")
    with gcol2:
        fig_g = go.Figure()
        for gt, col in [("AA", "관찰_AA"), ("Aa", "관찰_Aa"), ("aa", "관찰_aa")]:
            fig_g.add_trace(go.Scatter(x=df["세대"], y=df[col], name=f"{gt} 관찰",
                                       line=dict(color=COLOR[gt], width=3)))
        for gt, col in [("AA", "예상_AA"), ("Aa", "예상_Aa"), ("aa", "예상_aa")]:
            fig_g.add_trace(go.Scatter(x=df["세대"], y=df[col], name=f"{gt} 예상",
                                       line=dict(color=COLOR[gt], width=1.5, dash="dot"),
                                       opacity=0.6))
        fig_g.update_layout(
            title="② 유전자형 비율 (실선=관찰, 점선=예상)", height=340,
            plot_bgcolor=plot_bg, paper_bgcolor=plot_bg,
            yaxis=dict(range=[0, 1], title="비율"), xaxis=dict(title="세대"),
            legend=dict(orientation="h", y=1.18, font=dict(size=10)),
            margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_g, width='stretch')
        st.caption("실선과 점선이 겹치면 → 예상대로(p²·2pq·q²) 잘 맞는 거예요.")

    with st.expander("📊 표로 된 자료 보기 / 내려받기"):
        show_df = df.copy()
        for c in show_df.columns[1:]:
            show_df[c] = show_df[c].round(4)
        st.dataframe(show_df, hide_index=True, width='stretch')
        st.download_button("CSV로 내려받기",
                           show_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="simulation_data.csv", mime="text/csv")

# ---------------------------------------------------------------------
# 화면 3. 내 생각 정리
# ---------------------------------------------------------------------
with tab3:
    st.header("내가 찾은 평형 조건 ✍️")
    st.markdown(
        "충분히 실험해 봤다면, 아래에서 **평형에 꼭 필요할 것 같은 조건 5개**에 체크하세요. "
        "체크하지 않은 것은 '관련 없음'으로 봅니다."
    )
    letters = list(CONDITIONS.keys())
    cols = st.columns(2)
    for i, L in enumerate(letters):
        cols[i % 2].checkbox(f"조건 {L} — 평형에 필요할 것 같다", key=f"pick_{L}")
    picks = [L for L in letters if st.session_state.get(f"pick_{L}")]

    n = len(picks)
    if n == 5:
        st.success("좋아요! 5개를 골랐어요. **‘④ 결과 확인’** 탭에서 정답을 확인하세요. 🎉")
    elif n < 5:
        st.info(f"지금까지 {n}개 선택 — {5 - n}개 더 고르면 돼요.")
    else:
        st.warning(f"지금 {n}개 선택했어요. 5개만 남기고 줄여 보세요.")

    st.markdown("#### 관찰한 내용 적어보기 (탐구 질문)")
    st.text_area("1) 조건을 바꿨을 때 **유전자 비율(p, q)** 이 변한 조건은? 어떻게 변했나요?",
                 key="q1", height=80)
    st.text_area("2) 유전자 비율은 거의 그대로인데 **유전자형 비율(AA·Aa·aa)** 이 "
                 "예상값에서 벗어난 조건은?", key="q2", height=80)
    st.text_area("3) 그 변화가 **여러 세대 동안 계속** 이어졌나요? "
                 "(잠깐 출렁였다 돌아오는지 vs 계속 변하는지)", key="q3", height=80)
    st.text_area("4) 내가 고른 5개를 평형 조건이라고 생각한 **근거**(관찰한 그래프)는?",
                 key="q4", height=100)

# ---------------------------------------------------------------------
# 화면 4. 결과 확인
# ---------------------------------------------------------------------
with tab4:
    st.header("결과 확인 & 설명")
    picks = [L for L in CONDITIONS if st.session_state.get(f"pick_{L}")]
    reveal = teacher or st.button("🔓 정답 확인하기", type="primary")

    if not reveal:
        st.info("**‘③ 내 생각 정리’** 에서 5개를 고른 뒤, 위 **‘🔓 정답 확인하기’** "
                "버튼을 누르면 정답이 나와요.")
    else:
        chosen = set(picks)
        rows = []
        for L, info in CONDITIONS.items():
            picked = L in chosen
            correct = (picked == info["is_eq"])
            rows.append({
                "조건": L, "진짜 의미": info["real"],
                "평형 조건?": "⭕" if info["is_eq"] else "❌",
                "내 선택": "필요" if picked else "관련 없음",
                "채점": "정답 ✅" if correct else "다시 보기 🔁",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

        if picks:
            n_correct = sum(1 for r in rows if "정답" in r["채점"])
            st.metric("맞힌 개수", f"{n_correct} / 10")
            hit = sorted(chosen & set(EQ_LETTERS))
            miss = sorted(set(EQ_LETTERS) - chosen)
            trap = sorted(chosen & set(NEUTRAL_LETTERS))
            if hit:
                st.success("잘 찾았어요 👍 : " +
                           ", ".join(f"{L}({CONDITIONS[L]['real']})" for L in hit))
            if miss:
                st.warning("놓친 평형 조건 : " +
                           ", ".join(f"{L}({CONDITIONS[L]['real']})" for L in miss))
            if trap:
                st.error("함정에 걸렸어요 (평형 조건이 아니에요) : " +
                         ", ".join(f"{L}({CONDITIONS[L]['real']})" for L in trap))

        st.markdown("### 🔑 평형을 유지하는 5가지 조건")
        for L in EQ_LETTERS:
            st.markdown(f"- **{L} · {CONDITIONS[L]['real']}** — {CONDITIONS[L]['why']}")

        st.markdown("### ⚠️ 헷갈리기 쉬운 점 정리")
        st.markdown(
            """
- **처음 A 비율(F)** 은 시작값만 바꿔요. 평형이 *유지되는지*와는 상관없어요.
  조건만 다 맞으면 비율이 0.2든 0.8이든 그대로 유지돼요.
- **개체군이 작으면(A)** 자연선택이 없어도 **우연(유전적 부동)** 만으로 비율이 출렁여요.
  실험 번호를 바꿔 가며 확인해 보세요.
- **자연선택(E)** 은 유형별 *생존율 차이*로 특정 유전자를 늘리거나 줄여요.
- **돌연변이(C)·이주(D)** 는 유전자를 바꾸거나 바깥에서 들여와 비율을 바꿔요.
- **무작위 짝짓기(B)** 가 깨지면 **유전자 비율은 거의 그대로**여도
  유전자형 비율이 **예상값에서 벗어나요**(②탭의 '차이'가 가장 잘 드러나는 조건!).
- **세대 수(G)·움직임 속도(H)·배경(I)·색칠(J)** 은 보기/표시 방식일 뿐이에요.
            """
        )

st.sidebar.markdown("---")
st.sidebar.caption("고등학교 생명과학 · 하디-바인베르크 평형 탐구")
