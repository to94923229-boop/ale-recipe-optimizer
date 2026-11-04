"""
Al2O3 Thermal ALE (HF/TMA) — 물리 기반 EPC 모델과 최소 cycle time 계산

구조
----
1) epc()            : 공정 조건 → EPC (Å/cycle). calibrate.py가 문헌값에 맞춰 파라미터를 피팅
2) min_recipe()     : (T, P_HF, P_TMA, P_total) → 포화·퍼지 조건을 만족하는 최소 주입/퍼지 시간,
                      cycle time, EPC, throughput  (팀 노트: "시간은 출력으로 둔다")

물리 가정 (각 항의 근거 논문)
--------------------------------
노출량       X = a · P · t   (Torr·s). a = 장비별 노출 효율 (Z18 = 1 기준)
불화층       g_HF = s_HF · sqrt( 1 + c(T)·X/X0 ),  s_HF = 1 − exp(−X/X0)
             → 저노출에서는 Langmuir 표면 불화(선형 시작, Zywotko 분압 스캔),
               고노출에서는 포물선 확산(두께 ∝ √X, Cano 2019 식 8)으로 넘어가는 형태
             c(T) = c0·exp(−Ea/R·(1/T − 1/573K))  확산항의 열활성 (Hennessy 2017: 고온에서 불화 포화 없음)
TMA 제거율   r(T) = 1/(1+exp(−(T−T_r)/w_r))      (Lee 2016: 250°C 19% → 325°C 74%)
             정적 고노출 장비(Cano, Reif)는 제거율 곡선의 온도 이동 dTr을 따로 둔다
TMA 포화     s_TMA = 1 − exp(−X_TMA/X0_TMA)
AlF3 성장    G = G0·s_g(T)·s_HF·s_TMA,  s_g(T) = 1/(1+exp((T−T_g)/w_g))  저온일수록 AlF3 ALD 우세
                                                                     (DuMont 2017, Reif 2022, Zywotko 2018)
순 EPC       EPC = amp · A0 · g_HF · r(T) · s_TMA · L_ov − g · G
오버랩 손실  L_ov = 1 − k_ov·D_ov − k_res·D_res
             D = 섞여 들어간 전구체 도즈를 포화 노출량 X0로 나눈 값
             D_ov  : 펄스가 직접 겹친 시간 × (P/X0)        (Zywotko 2018 오버랩 실험)
             D_res : 퍼지 뒤 잔류분 τ·exp(−tp/τ) × (P/X0)  (Reif 2022 퍼지 스캔)
             HF 후 퍼지는 벽 흡착 때문에 별도의 τ_HF
             이 손실을 퍼지당 2.5%로 제한하는 최소 퍼지:
                 tp_min = τ·ln( k_res·(P/X0)·τ / 0.025 )   ← 팀 노트의 τ·ln(C0/C_th), C0 ∝ P
             → 팀 노트의 t_purge ≈ τ·ln(C0/C_th) 와 같은 형태

장비 가정 (팀 노트 4장)
------------------------
N2 질량유량 고정, 압력은 배기 밸브로 조절 → 체적유량 ∝ 1/P → τ = V/Q ∝ P_total
"""
from __future__ import annotations

import sys
try:                                     # Windows 콘솔에서 Å, ≥, ✔ 같은 문자가 깨지지 않게
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import json
from pathlib import Path
import numpy as np

R_GAS = 8.314
PARAM_FILE = Path(__file__).parent / "params.json"
PAPERS = ["Z18", "L16", "C19", "R22"]          # Z18 = 노출량 기준 장비 (a = 1)

# ---------------------------------------------------------------- 파라미터 정의
# name: (초기값, 하한, 상한, 설명)
SHARED = {
    "lnX0_HF":  (np.log(0.08), np.log(1e-3), np.log(5.0), "HF 표면 불화 특성 노출량 ln(Torr·s)"),
    "lnX0_TMA": (np.log(0.04), np.log(1e-3), np.log(5.0), "TMA 포화 특성 노출량 ln(Torr·s)"),
    "lnc0":     (np.log(0.05), np.log(1e-5), np.log(5.0), "불화 확산(포물선)항 가중치 @300°C"),
    "Ea":       (60e3, 0.0, 200e3, "확산항 활성화에너지 J/mol"),
    "A0":       (1.0, 0.1, 5.0, "식각 진폭 Å/cycle"),
    "T_r":      (270.0, 180.0, 340.0, "TMA 제거율 시그모이드 중심 °C"),
    "w_r":      (25.0, 3.0, 80.0, "TMA 제거율 시그모이드 폭 °C"),
    "G0":       (0.6, 0.0, 1.5, "AlF3 ALD 최대 성장 Å/cycle"),
    "T_g":      (250.0, 150.0, 330.0, "AlF3 성장 감소 중심 °C"),
    "w_g":      (20.0, 3.0, 80.0, "AlF3 성장 감소 폭 °C"),
    "k_ov":     (1.0, 0.1, 5.0, "직접 오버랩 손실 계수 (펄스끼리 겹침)"),
    "k_res":    (0.5, 0.01, 20.0, "잔류 전구체 손실 계수 (퍼지 부족, k_ov × 잔류 초기농도비)"),
    "ln_tau_Z18": (np.log(0.05), np.log(1e-3), np.log(5.0), "Z18 장비 τ ln(s)"),
    "ln_tau_R22": (np.log(1.5), np.log(0.05), np.log(20.0), "R22 장비 TMA 퍼지 τ ln(s) @0.75 Torr"),
    "ln_tauHF_R22": (np.log(4.0), np.log(0.05), np.log(40.0), "R22 장비 HF 퍼지 τ ln(s) @0.75 Torr (기상+벽 탈착)"),
    "beta_tau": (0.5, 0.0, 1.0, "R22 장비 τ의 압력 지수: τ ∝ (P_total/0.75)^β  (질량유량 고정이면 β = 1)"),
}
# 장비별 보정계수 (Z18 = 0 고정). ln 스케일, 약한 사전분포(σ=PRIOR_SD)로 0 쪽으로 당김
PER_PAPER = {
    "ln_aHF":  (0.0, -8.0, 6.0, "HF 노출 효율"),
    "ln_aTMA": (0.0, -8.0, 6.0, "TMA 노출 효율"),
    "ln_amp":  (0.0, -1.5, 1.5, "식각 진폭 배율"),
    "ln_g":    (0.0, -6.0, 1.5, "AlF3 성장 배율"),
    "dTr":     (0.0, -4.0, 4.0, "TMA 제거율 곡선의 온도 이동 (값 × 25°C, 음수 = 더 낮은 온도에서 제거)"),
}
DTR_UNIT = 25.0   # dTr 1 = 25°C
PRIOR_SD = 2.0


def param_names() -> list[str]:
    names = list(SHARED)
    for p in PAPERS[1:]:
        names += [f"{k}@{p}" for k in PER_PAPER]
    return names


def bounds_and_init():
    x0, lo, hi = [], [], []
    for k, (v, l, h, _) in SHARED.items():
        x0.append(v); lo.append(l); hi.append(h)
    for p in PAPERS[1:]:
        for k, (v, l, h, _) in PER_PAPER.items():
            x0.append(v); lo.append(l); hi.append(h)
    return np.array(x0), np.array(lo), np.array(hi)


def unpack(theta) -> dict:
    return dict(zip(param_names(), np.asarray(theta, float)))


def load_params(path: Path = PARAM_FILE) -> dict:
    return json.loads(Path(path).read_text())["params"]


def load_meta(path: Path = PARAM_FILE) -> dict:
    return json.loads(Path(path).read_text())["meta"]


# ---------------------------------------------------------------- 물리 모델
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def paper_factor(p: dict, paper: str, key: str) -> float:
    return 0.0 if paper == "Z18" else p.get(f"{key}@{paper}", 0.0)


def tau_of(p: dict, paper: str, P_total=None, beta=None):
    """퍼지 특성시간 (τ_TMA, τ_HF) (s).
    Z18: 점성흐름 장비, 측정된 HF 벽 효과 없음 → 두 값 동일.
    그 외(R22 기준): τ ∝ (P_total/0.75)^β, HF는 벽 흡착 때문에 별도의 τ.
    beta=None 이면 피팅값 사용 (문헌 장비), 팀 장비 가정(질량유량 고정)은 beta=1."""
    Pt = np.asarray(P_total if P_total is not None else 0.75, float)
    if paper == "Z18":
        t = np.exp(p["ln_tau_Z18"]) * np.ones_like(Pt)
        return t, t
    b = p.get("beta_tau", 1.0) if beta is None else beta
    scale = (Pt / 0.75) ** b
    return np.exp(p["ln_tau_R22"]) * scale, np.exp(p["ln_tauHF_R22"]) * scale


def _overlap(tp):
    return np.maximum(-np.asarray(tp, float), 0.0)


def _residual(tp, tau):
    """퍼지 tp 뒤 남은 전구체를 적분한 등가 시간: ∫ exp(−t/τ) dt (tp부터) = τ·exp(−tp/τ)"""
    return tau * np.exp(-np.maximum(np.asarray(tp, float), 0.0) / tau)


def _loss(p, rate_HF, rate_TMA, tp_HF, tp_TMA, tau, tauHF):
    """잔류·오버랩 전구체에 의한 식각 손실 비율.
    rate = a·P/X0 (1/s): 해당 전구체가 1초 동안 주는 '포화 노출량 단위' 도즈.
    HF 펄스 끝이 TMA와 겹치면(tp_HF<0) 겹친 시간 동안 두 전구체가 동시에 존재한다.
    퍼지 뒤 남은 전구체는 τ·exp(−tp/τ) 만큼의 등가 시간으로 다음 펄스에 섞인다.
    → 손실 ∝ 섞인 전구체의 양 (팀 노트 퍼지 모델의 C0 ∝ P 와 같은 의미)"""
    D_ov = (rate_HF + rate_TMA) * (_overlap(tp_HF) + _overlap(tp_TMA))
    D_res = rate_HF * _residual(tp_HF, tauHF) + rate_TMA * _residual(tp_TMA, tau)
    o = p["k_ov"] * D_ov + p["k_res"] * D_res
    return 1.0 - o, o


def epc(p: dict, T, P_HF, P_TMA, t_HF, t_TMA, tp_HF, tp_TMA, paper="Z18", P_total=None,
        return_parts=False):
    """공정 조건 → EPC (Å/cycle). 배열 입력 가능."""
    T = np.asarray(T, float); TK = T + 273.15
    aHF = np.exp(paper_factor(p, paper, "ln_aHF"))
    aTMA = np.exp(paper_factor(p, paper, "ln_aTMA"))
    amp = np.exp(paper_factor(p, paper, "ln_amp"))
    gmul = np.exp(paper_factor(p, paper, "ln_g"))

    X0h, X0t = np.exp(p["lnX0_HF"]), np.exp(p["lnX0_TMA"])
    tp_HF = np.asarray(tp_HF, float); tp_TMA = np.asarray(tp_TMA, float)
    t_HF = np.asarray(t_HF, float); t_TMA = np.asarray(t_TMA, float)
    # 오버랩 구간은 순수 노출에서 빠짐 (HF 펄스 끝이 TMA와 겹침)
    ov = np.maximum(-tp_HF, 0.0)
    X_HF = aHF * np.asarray(P_HF, float) * np.maximum(t_HF - ov, 0.0)
    X_TMA = aTMA * np.asarray(P_TMA, float) * np.maximum(t_TMA - ov, 0.0)

    c = np.exp(p["lnc0"]) * np.exp(-p["Ea"] / R_GAS * (1.0 / TK - 1.0 / 573.15))
    s_HF = 1.0 - np.exp(-X_HF / X0h)
    g_HF = s_HF * np.sqrt(1.0 + c * X_HF / X0h)
    s_TMA = 1.0 - np.exp(-X_TMA / X0t)
    r = _sig((T - p["T_r"] + DTR_UNIT * paper_factor(p, paper, "dTr")) / p["w_r"])

    tau, tauHF = tau_of(p, paper, P_total)
    L, o_eff = _loss(p, aHF * np.asarray(P_HF, float) / X0h, aTMA * np.asarray(P_TMA, float) / X0t,
                     tp_HF, tp_TMA, tau, tauHF)

    E = amp * p["A0"] * g_HF * r * s_TMA * L
    G = gmul * p["G0"] * _sig(-(T - p["T_g"]) / p["w_g"]) * s_HF * s_TMA
    out = E - G
    if return_parts:
        return dict(EPC=out, etch=E, growth=G, s_HF=s_HF, s_TMA=s_TMA, g_HF=g_HF, r=r, L_ov=L, o_eff=o_eff)
    return out


# ---------------------------------------------------------------- 최소 레시피 (출력 Y)
SAT_LEVEL = 0.95      # 포화 판정: 표면 반응 95%
PURGE_LOSS = 0.05     # 퍼지 판정: 잔류 전구체에 의한 EPC 손실 합계 5% 이하 (퍼지당 2.5%)
REF_PAPER = "R22"     # 가상 기준 장비: 퍼지(τ) 특성은 R22, 노출량은 실제 분압 기준(a = 1)
Z18_P_TOTAL = 1.98    # Zywotko 총압 = N2 1.5 + HF 0.32 + TMA 0.16 Torr (팀 결정 4) — Z18 τ의 기준 압력
N2_MIN = 0.5          # 유동식 장비의 최소 N2 배경압 (Torr). 문헌 장비 1–1.5 Torr의 절반, 설정값


def min_recipe(p: dict, T, P_HF, P_TMA, P_total, sat=SAT_LEVEL, purge_loss=PURGE_LOSS,
               tau_scale=1.0, Ea_wall=0.0, beta=1.0, ref=REF_PAPER, m_HF=1.0):
    """
    (T, P_HF, P_TMA, P_total) → 최소 주입/퍼지 시간과 그때의 EPC, cycle time, throughput
    모든 입력 Torr / °C. 배열 브로드캐스팅 가능.
    tau_scale: 장비 τ 민감도 분석용 배율 (1 = R22 기준)
    beta     : τ ∝ (P_total/0.75)^β. 기본 1 = 팀 장비 가정(N2 질량유량 고정 → τ = V/Q ∝ P)
    Ea_wall  : HF 벽 탈착 활성화에너지 (J/mol). 문헌에 온도별 퍼지 데이터가 없어 기본 0(온도 무관).
               팀 노트 "저온에서는 T가 퍼지의 주 인자일 수 있다"를 민감도 분석할 때 값을 넣는다.
               (300°C 기준으로 τ_HF·exp(Ea/R·(1/T − 1/573K)) — 기판 온도를 벽 온도의 대리값으로 사용)
    ref      : 퍼지 τ를 가져올 장비. "R22"(정적 노출, HF 벽 효과 포함) 또는
               "Z18"(점성흐름, 1.95 Torr에서 피팅된 τ를 (P_total/1.95)^β로 환산)
    m_HF     : ref="Z18"일 때 HF 퍼지 τ 배율. Zywotko에는 HF 벽 효과를 분리할 데이터가 없어 기본 1.
    """
    T, P_HF, P_TMA, P_total = np.broadcast_arrays(*(np.asarray(v, float) for v in (T, P_HF, P_TMA, P_total)))
    X0h, X0t = np.exp(p["lnX0_HF"]), np.exp(p["lnX0_TMA"])
    nsat = -np.log(1.0 - sat)
    t_HF = nsat * X0h / P_HF            # 최소 주입시간 ≈ 포화 노출량 / P   (팀 노트 2장)
    t_TMA = nsat * X0t / P_TMA

    if ref == "Z18":
        tau = np.exp(p["ln_tau_Z18"]) * (P_total / Z18_P_TOTAL) ** beta
        tauHF = tau * m_HF
    else:
        tau, tauHF = tau_of(p, "R22", P_total, beta=beta)
    tau = tau * tau_scale
    tauHF = tauHF * tau_scale * np.exp(Ea_wall / R_GAS * (1.0 / (T + 273.15) - 1.0 / 573.15))
    # 두 퍼지가 손실 예산을 반씩 나눠 쓴다:  k_res·(P/X0)·τ·exp(−tp/τ) = loss/2
    #  → tp = τ·ln( C0 / C_th ),  C0/C_th = k_res·(P/X0)·τ / (loss/2)   (팀 노트 퍼지 모델)
    half = purge_loss / 2.0
    tp_HF = tauHF * np.log(np.maximum(p["k_res"] * P_HF / X0h * tauHF / half, 1.0))
    tp_TMA = tau * np.log(np.maximum(p["k_res"] * P_TMA / X0t * tau / half, 1.0))

    # 노출량 기준 장비(a=1) + R22 퍼지 특성으로 EPC 계산
    parts = _epc_ref(p, T, P_HF, P_TMA, t_HF, t_TMA, tp_HF, tp_TMA, tau, tauHF)
    cycle = t_HF + tp_HF + t_TMA + tp_TMA
    return dict(T=T, P_HF=P_HF, P_TMA=P_TMA, P_total=P_total,
                t_HF=t_HF, tp_HF=tp_HF, t_TMA=t_TMA, tp_TMA=tp_TMA,
                t_react=t_HF + t_TMA, t_purge=tp_HF + tp_TMA, cycle_time=cycle,
                EPC=parts["EPC"], throughput_A_per_min=parts["EPC"] / cycle * 60.0,
                tau=tau, tau_HF=tauHF)


def _epc_ref(p, T, P_HF, P_TMA, t_HF, t_TMA, tp_HF, tp_TMA, tau, tauHF):
    """기준 장비용 EPC: 노출 효율 a = 1, 진폭·성장 배율 = 1 (Z18 화학), τ는 외부에서 지정."""
    TK = T + 273.15
    X0h, X0t = np.exp(p["lnX0_HF"]), np.exp(p["lnX0_TMA"])
    X_HF, X_TMA = P_HF * t_HF, P_TMA * t_TMA
    c = np.exp(p["lnc0"]) * np.exp(-p["Ea"] / R_GAS * (1.0 / TK - 1.0 / 573.15))
    s_HF = 1.0 - np.exp(-X_HF / X0h)
    g_HF = s_HF * np.sqrt(1.0 + c * X_HF / X0h)
    s_TMA = 1.0 - np.exp(-X_TMA / X0t)
    r = _sig((T - p["T_r"]) / p["w_r"])
    L, _ = _loss(p, P_HF / X0h, P_TMA / X0t, tp_HF, tp_TMA, tau, tauHF)
    E = p["A0"] * g_HF * r * s_TMA * L
    G = p["G0"] * _sig(-(T - p["T_g"]) / p["w_g"]) * s_HF * s_TMA
    return dict(EPC=E - G, etch=E, growth=G)


if __name__ == "__main__":
    p = load_params()
    out = min_recipe(p, T=300, P_HF=0.32, P_TMA=0.16, P_total=0.75)
    for k, v in out.items():
        print(f"{k:22s} {float(v):.4g}")
