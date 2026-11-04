"""
Al₂O₃ HF/TMA Thermal ALE 레시피 최적화 웹 데모 (Streamlit)
실행:  streamlit run app.py
"""
import warnings
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
import core  # noqa: E402

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"

st.set_page_config(page_title="Al₂O₃ Thermal ALE 레시피 최적화", page_icon="🧪", layout="wide")

C_BASE, C_IN, C_A, C_B = "#6b7280", "#0ea5e9", "#059669", "#7c3aed"


@st.cache_resource
def _load():
    return core.load()


@st.cache_data(show_spinner=False)
def _run(T, PH, PT, Pt, tmax, epc_min, tau):
    p, sur = _load()
    return core.run(core.make_cfg(T, PH, PT, Pt, tmax, epc_min, tau), p, sur)


@st.cache_data(show_spinner=False)
def _sweep_tmax(epc_min, tau, loss):
    return core.sweep_tmax(_load()[0], epc_min, tau, loss, temps=range(250, 351, 5))


@st.cache_data(show_spinner=False)
def _sweep_tau(tmax, epc_min, loss):
    return core.sweep_tau(_load()[0], tmax, epc_min, loss)


@st.cache_data(show_spinner=False)
def _sweep_epc(tmax, tau, loss):
    tg = (0.3, 0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3)
    return core.sweep_epc_target(_load()[0], tmax, tau, loss, targets=tg)


@st.cache_data(show_spinner=False)
def _sweep_loss(tmax, epc_min):
    return core.sweep_loss(_load()[0], tmax, epc_min)


# ============================================================ 사이드바 입력
st.sidebar.title("공정 입력")
st.sidebar.caption("공정 변수 4개 (T, P_HF, P_TMA, P_total)")
T = st.sidebar.number_input("T — 기판 온도 (°C)", 200.0, 350.0, 300.0, 5.0)
PH = st.sidebar.number_input("P_HF — HF 분압 (Torr)", 0.02, 1.0, 0.08, 0.01, format="%.3f")
PT = st.sidebar.number_input("P_TMA — TMA 분압 (Torr)", 0.02, 1.0, 0.04, 0.01, format="%.3f")
Pt = st.sidebar.number_input("P_total — 총압 (Torr)", 0.5, 3.5, 1.12, 0.05, format="%.2f",
                             help="P_total = N₂ 배경압 + P_HF + P_TMA (유동식 정의). 모델 A는 이 값을 고정합니다.")
st.sidebar.divider()
st.sidebar.caption("공정 제약과 장비 시나리오")
tmax = st.sidebar.slider("T_max — 허용 최고 온도 (°C)", 250, 350, 300, 5)
epc_min = st.sidebar.slider("EPC 목표 (Å/cycle) 이상", 0.2, 1.3, 0.90, 0.05)
tau = st.sidebar.select_slider("장비 퍼지 시나리오 (τ 배율)", options=[1, 3, 10, 30, 100], value=1,
                               format_func=lambda v: f"×{v}")
st.sidebar.caption(core.TAU_LABEL[tau])
st.sidebar.info("τ는 장비(반응기 부피/유량)로 정해지는 값이라 우리 장비의 실측값은 아직 없습니다. "
                "결과를 볼 때 시나리오별로 비교하세요.")

cfg = core.make_cfg(T, PH, PT, Pt, tmax, epc_min, tau)
errs = core.validate_inputs(cfg)

st.title("Al₂O₃ Thermal ALE 레시피 최적화")
st.caption("HF–TMA sequential thermal ALE · 머신러닝 대리모델 + 물리 기반 최적화 · "
           "합성 데이터(논문 anchor 보정) 기반 데모이며 실제 장비 검증 전의 예측입니다.")

tab1, tab2, tab3 = st.tabs(["① 레시피 최적화", "② Trade-off 분석", "③ 모델 근거와 한계"])

# ============================================================ ① 레시피 최적화
with tab1:
    if errs:
        for e in errs:
            st.error(e)
        st.stop()

    with st.spinner("모델 A / B 최적화 중 (약 1–3초)..."):
        R = _run(T, PH, PT, Pt, tmax, epc_min, tau)
    base, inp, A, B = R["base"], R["input"], R["A"], R["B"]

    if B["ok"]:
        st.success(f"모델 B 최적 레시피: 총 cycle time **{B['cycle']:.2f} s** · EPC **{B['EPC']:.2f} Å/cycle** · "
                   f"Throughput **{B['throughput']:.1f} Å/min** (Lee 2016 기준 {base['cycle']:.0f} s · "
                   f"{base['throughput']:.2f} Å/min, EPC 조건이 달라 직접 비교는 주의) — 모든 공정 제약 충족")
        if B.get("extrapolated"):
            st.warning("이 최적점은 문헌 분압 범위(0.04–8 Torr) 밖입니다 (외삽). 값의 불확실성이 큽니다.")
    else:
        bad = [c[0] for c in B["checks"] if not c[1]]
        reason = ""
        if not B["checks"][0][1]:
            reason = (f" T_max {tmax}°C에서 낼 수 있는 EPC는 약 {core.epc_ceiling(_load()[0], tmax):.2f} Å/cycle입니다. "
                      "T_max를 올리거나 EPC 목표를 낮추세요.")
        st.warning(f"모델 B가 모든 제약을 만족하지 못했습니다 ({', '.join(bad)}).{reason}")
    st.caption(f"장비 시나리오: {core.TAU_LABEL[tau]} · 포화 95% · 잔류 손실 ≤ 5% · N₂ 배경압 ≥ {cfg.n2_min} Torr")

    def row(r, key, fmt):
        return fmt.format(r[key]) if r is not None else "—"

    cols = [("기준 레시피\n(Lee 2016)", base), ("입력 레시피\n(최소 시간)", inp),
            ("모델 A\n(반응시간 최소)", A), ("모델 B\n(총 cycle 최소)", B)]
    rows = {
        "T (°C)": lambda r: f"{r['T']:.1f}",
        "P_HF / P_TMA (Torr)": lambda r: f"{r['P_HF']:.3f} / {r['P_TMA']:.3f}",
        "P_total (N₂) (Torr)": lambda r: f"{r['P_total']:.2f} ({r['P_N2']:.2f})",
        "① HF / TMA Exposure (s)": lambda r: f"{r['t_HF']:.2f} / {r['t_TMA']:.2f}",
        "② Purge HF후 / TMA후 (s)": lambda r: f"{r['tp_HF']:.2f} / {r['tp_TMA']:.2f}",
        "③ 총 Cycle Time (s)": lambda r: f"{r['cycle']:.2f}",
        "④ EPC (Å/cycle)": lambda r: f"{r['EPC_lit']:.2f} (측정)" if "EPC_lit" in r else f"{r['EPC']:.3f}",
        "Throughput (Å/min)": lambda r: f"{r['throughput']:.2f}",
        "반응 : 퍼지 (s)": lambda r: f"{r['t_react']:.2f} : {r['t_purge']:.2f}",
        "공정 제약 충족": lambda r: ("—" if "ok" not in r else ("✔ 모두 충족" if r["ok"] else "✘ 미충족") + (" (외삽)" if r.get("extrapolated") else "")),
    }
    df = pd.DataFrame({name.replace("\n", " "): [f(r) if r is not None else "—" for f in rows.values()]
                       for name, r in cols}, index=list(rows.keys()))
    st.dataframe(df, use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("반응시간 – Purge Trade-off")
        names, react, purge = [], [], []
        for n, r in (("기준", base), ("입력", inp), ("모델 A", A), ("모델 B", B)):
            if r is not None:
                names.append(n); react.append(r["t_react"]); purge.append(r["t_purge"])
        fig = go.Figure()
        fig.add_bar(x=names, y=react, name="반응(Exposure) 시간", marker_color="#0ea5e9")
        fig.add_bar(x=names, y=purge, name="Purge 시간", marker_color="#f59e0b")
        fig.update_layout(barmode="stack", yaxis_title="시간 (s)", height=380, margin=dict(t=20, b=20),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("공정 제약 체크")
        pick = st.radio("대상", ["모델 B", "모델 A", "입력 레시피"], horizontal=True)
        tgt = {"모델 B": B, "모델 A": A, "입력 레시피": inp}[pick]
        if tgt is None:
            st.info("P_total이 작아 모델 A를 계산할 수 없습니다 (P_total − N₂ 최소 ≥ 0.04 Torr 필요).")
        else:
            for name, ok, desc in tgt["checks"]:
                st.markdown(f"{'✅' if ok else '❌'} **{name}** — {desc}")
            if not tgt["checks"][-1][1]:
                st.caption("문헌 분압 범위 밖이면 외삽이라 불확실성이 큽니다.")
            if "EPC_ml" in tgt:
                st.caption(f"대리모델(ML) 예측 EPC {tgt['EPC_ml']:.3f} Å/cycle · 괄호 밖 값은 물리 생성식으로 다시 계산한 값")

    with st.expander("이 결과를 읽을 때 주의할 점"):
        st.markdown(
            "- 기준 레시피(Lee 2016, EPC 0.46 Å/cycle)와 최적 레시피(EPC 약 1.0)는 노출량이 달라서, "
            "'cycle time 감소율'은 같은 식각량 기준 비교가 아닙니다. Throughput(Å/min)을 함께 보세요.\n"
            "- 퍼지 시간은 장비 τ 시나리오에 크게 좌우됩니다. 사이드바에서 τ를 바꿔 보세요.\n"
            "- 최적점이 N₂ 최소압, 잔류 손실 5% 같은 제약 경계에 붙는 경우가 많습니다.")

# ============================================================ ② Trade-off
with tab2:
    st.markdown("사이드바의 T_max, EPC 목표, τ 시나리오를 기준으로 **물리 생성식으로 빠르게 다시 최적화**한 결과입니다 "
                "(모델 B와 같은 문제, 대리모델 없이 계산). 허용 손실을 바꾸는 분석은 이 방식으로만 가능합니다.")

    st.subheader("1) 장비 퍼지 τ에 따른 최소 cycle time과 최적 압력")
    S = _sweep_tau(tmax, epc_min, 0.05)
    ok = [s for s in S]
    fig = go.Figure()
    fig.add_bar(x=[f"×{s['tau_scale']}" for s in ok], y=[s["t_react"] for s in ok], name="반응 시간", marker_color="#0ea5e9")
    fig.add_bar(x=[f"×{s['tau_scale']}" for s in ok], y=[s["t_purge"] for s in ok], name="퍼지 시간", marker_color="#f59e0b")
    fig.add_scatter(x=[f"×{s['tau_scale']}" for s in ok], y=[s["P_total"] for s in ok], name="최적 P_total (Torr)",
                    yaxis="y2", mode="lines+markers", line=dict(color="#111827"))
    fig.update_layout(barmode="stack", height=380, margin=dict(t=20, b=20), xaxis_title="장비 τ 배율",
                      yaxis_title="최소 cycle time (s)",
                      yaxis2=dict(title="P_total (Torr)", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)
    if not all(s["feasible"] for s in S):
        st.warning("일부 τ에서 EPC 목표를 만족하지 못했습니다 (이 경우 cycle 값은 의미 없음).")
    st.caption("τ가 커질수록 퍼지가 길어지고, 최적 압력은 낮아집니다 (낮은 압력 = 퍼지가 짧아짐).")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("2) T_max에 따라 낼 수 있는 최대 EPC")
        rows_ = _sweep_tmax(epc_min, tau, 0.05)
        fig = go.Figure()
        fig.add_scatter(x=[r["tmax"] for r in rows_], y=[r["EPC_max"] for r in rows_], mode="lines+markers",
                        name="T_max에서 낼 수 있는 최대 EPC", line=dict(color="#111827"))
        fig.add_hline(y=epc_min, line_dash="dot", line_color="#ef4444", annotation_text="EPC 목표")
        fig.update_layout(height=380, margin=dict(t=20, b=20), xaxis_title="T_max (°C)", yaxis_title="EPC (Å/cycle)",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
        inf = [r["tmax"] for r in rows_ if not r["feasible"]]
        feas = [r for r in rows_ if r["feasible"]]
        if feas:
            st.caption(f"목표를 낼 수 있는 T_max에서는 최소 cycle이 거의 일정합니다 (약 {feas[0]['cycle']:.2f} s). "
                       "T_max는 cycle time보다 '목표 달성 가능 여부'를 결정합니다.")
        if inf:
            st.caption(f"T_max ≤ {max(inf):.0f}°C에서는 EPC 목표 {epc_min:.2f}를 낼 수 없습니다 "
                       "(포화 노출에서 EPC는 온도로 정해지는 모델 구조 때문).")
    with c2:
        st.subheader("3) EPC 목표에 따른 최소 cycle (Pareto)")
        fig = go.Figure()
        for tm_, col in ((290, "#ef4444"), (300, "#f59e0b"), (310, C_A), (325, C_B)):
            rr = [r for r in _sweep_epc(tm_, tau, 0.05) if r["feasible"]]
            fig.add_scatter(x=[r["target"] for r in rr], y=[r["cycle"] for r in rr], mode="lines+markers",
                            name=f"T_max {tm_}°C", line=dict(color=col))
        fig.update_layout(height=380, margin=dict(t=20, b=20), xaxis_title="EPC 목표 (Å/cycle)",
                          yaxis_title="최소 cycle time (s)", yaxis_type="log", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"현재 장비 시나리오 {core.TAU_LABEL[tau].split(' ')[0]} 기준. T_max가 EPC 목표를 막기 시작하는 지점에서 "
                   "cycle이 급격히 늘고, 그 너머는 도달 불가라 선이 끝납니다.")

    st.subheader("4) 퍼지 기준(허용 EPC 손실)에 따른 최소 cycle (s)")
    L = _sweep_loss(tmax, epc_min)
    tab = pd.DataFrame([{"τ 배율": f"×{r['tau_scale']}", "허용 손실": f"{r['loss']:.0%}", "cycle (s)": r["cycle"],
                         "퍼지 비중": r["purge_share"], "feasible": r["feasible"]} for r in L])
    piv = tab.pivot(index="τ 배율", columns="허용 손실", values="cycle (s)")
    piv = piv.reindex(index=[f"×{t}" for t in (1, 10, 100)], columns=["1%", "2%", "5%", "10%"])
    st.dataframe(piv.style.format("{:.2f}"), use_container_width=True)
    st.caption("퍼지는 '잔류·겹침 전구체로 식각량이 허용 손실 넘게 줄지 않는 최소 시간'으로 정의했습니다 "
               "(퍼지 시간 = τ·ln(C₀/C_th)). 허용 손실을 엄격하게 잡을수록 퍼지가 길어집니다. "
               "겹침 실험 근거는 Zywotko 2018 (300°C 한 조건)이라 다른 온도는 가정입니다.")

# ============================================================ ③ 모델 근거와 한계
with tab3:
    st.subheader("파이프라인")
    st.markdown(
        "1. **물리식 보정** — 논문 anchor 39행(Lee 2016, Zywotko 2018, Cano 2019)으로 불화·TMA 제거·AlF₃ 성장·퍼지 손실 항을 피팅\n"
        "2. **합성 데이터** — 보정된 식으로 20,000점 생성 (T, P_HF, P_TMA, P_total, 장비 τ)\n"
        "3. **대리모델** — GP, HistGB, RandomForest, MLP를 5-fold로 비교해 목표별 최적 모델 선택\n"
        "4. **제약 최적화** — 모델 A(반응시간 최소, P_total·퍼지 고정) → 모델 B(총 cycle 최소, P_total 포함)\n"
        "5. **재검증** — ML이 찾은 조건을 물리식으로 다시 계산해 제약 확인")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("보정 정확도")
        fs = pd.read_csv(ASSETS / "fit_summary.csv")
        fs.columns = ["논문", "실험", "n", "RMSE (Å/cycle)", "MAE (Å/cycle)"]
        st.dataframe(fs.style.format({"n": "{:.0f}", "RMSE (Å/cycle)": "{:.3f}", "MAE (Å/cycle)": "{:.3f}"}),
                     use_container_width=True, hide_index=True, height=500)
        rt = json.loads((ASSETS / "reif_transfer.json").read_text(encoding="utf-8"))
        st.metric("Reif 2022 전이 검증 RMSE (학습에서 제외한 데이터)", f"{rt['rmse_transfer']:.2f} Å/cycle",
                  help=f"Reif를 포함해 다시 피팅하면 {rt['rmse_fullfit']:.2f}")
    with c2:
        st.subheader("대리모델 비교 (5-fold CV)")
        cv = pd.read_csv(ASSETS / "train_cv.csv")
        cv = cv[["model", "target", "R2", "MAE"]]
        cv.columns = ["모델", "목표", "R²", "MAE"]
        st.dataframe(cv.style.format({"R²": "{:.4f}", "MAE": "{:.4f}"}), use_container_width=True, hide_index=True, height=340)
        st.caption("R²가 높은 것은 대리모델이 '생성식'을 잘 흉내 낸다는 뜻이며, 식이 실제 공정을 잘 맞춘다는 뜻은 아닙니다. "
                   "실제 공정과의 일치는 위 보정 RMSE와 Reif 전이 검증으로 판단하세요.")

    st.subheader("가정과 한계 (결과 해석 시 함께 말할 것)")
    st.markdown(
        "1. **장비 퍼지 τ가 결과를 크게 좌우합니다.** Zywotko 장비(τ×1)는 퍼지가 거의 필요 없고, Reif형(τ×100)은 최소 cycle이 약 1 s → 28 s로 늘어납니다. 우리 장비의 τ는 미정입니다.\n"
        "2. **τ ∝ P_total은 장비 가정**(N₂ 질량유량 고정)입니다. Reif 데이터는 이 관계와 맞지 않았습니다.\n"
        "3. **포화 노출에서 EPC는 온도로만 정해집니다.** 그래서 압력으로 저온을 보상할 수 없고, T_max가 보상 한계 온도(기본 설정에서 약 294°C)보다 낮으면 EPC 목표를 낼 수 없습니다.\n"
        "4. **Reif 2022는 팀 결정으로 보정에서 제외**했고, Cano 200·250°C는 오차가 커서 보조 데이터로만 씁니다.\n"
        "5. **HF 벽 탈착의 온도 의존성은 데이터가 없습니다.**\n"
        "6. 최적점이 문헌 분압 범위(0.04–8 Torr) 밖이면 '외삽'으로 표시합니다.\n"
        "7. 합성 데이터 기반이므로 실제 장비 검증이 필요합니다.")

    pdf = ASSETS / "calibration.pdf"
    if pdf.exists():
        st.download_button("보정 검증 리포트(PDF) 내려받기", pdf.read_bytes(), "calibration.pdf", "application/pdf")
    imgs = [("dataset_overview.png", "합성 데이터 개요"), ("train_report.png", "대리모델 학습 결과")]
    for f, cap in imgs:
        if (ASSETS / f).exists():
            with st.expander(cap):
                st.image(str(ASSETS / f))

st.divider()
st.caption("CHEMIC · Al₂O₃ Thermal ALE ML Recipe Optimization · 참고: Lee 2016, Zywotko 2018, Cano 2019, Reif 2022 외")
