"""
웹 데모용 계산 모듈 — optimize.py(모델 A/B)를 함수 하나로 호출할 수 있게 감싼 것.
원본 ale_ml 의 generator.py / train.py / optimize.py 는 수정하지 않고 그대로 사용한다.

- run(...)            : 입력 레시피 → 기준 / 입력 / 모델 A / 모델 B 결과 + 제약 체크 (ML 대리모델 + 물리식 재검증)
- physics_optimum(...) : 물리 생성식만으로 모델 B 최적점 탐색 (빠름, 민감도 분석용. 허용 손실·τ를 바꿀 수 있음)
- sweeps              : T_max / 장비 τ / EPC 목표 / 허용 손실에 따른 최소 cycle time
"""
from types import SimpleNamespace
import numpy as np
import generator as G
import optimize as O

TAU_LABEL = {1: "τ×1 (Zywotko형 · 퍼지 거의 불필요)", 3: "τ×3", 10: "τ×10", 30: "τ×30",
             100: "τ×100 (Reif형 · 퍼지 필요)"}
NSAT = -np.log(1.0 - G.SAT_LEVEL)


def load():
    """보정 파라미터와 대리모델 (앱에서는 한 번만 불러오도록 캐시해서 사용)"""
    return G.load_params(), O.Surrogate()


def make_cfg(T, P_HF, P_TMA, P_total, tmax, epc_min, tau_scale, n2_min=G.N2_MIN, tmin=200.0):
    return SimpleNamespace(T=T, P_HF=P_HF, P_TMA=P_TMA, P_total=P_total, tmin=tmin, tmax=tmax,
                           epc_min=epc_min, n2_min=n2_min, tau_scale=tau_scale)


def validate_inputs(cfg):
    """입력 오류 메시지 리스트 (비어 있으면 정상)"""
    msg = []
    lo, hi = O.P_BOUNDS
    if not (lo <= cfg.P_HF <= hi):
        msg.append(f"P_HF는 {lo}–{hi} Torr 범위여야 합니다 (학습 데이터 범위).")
    if not (lo <= cfg.P_TMA <= hi):
        msg.append(f"P_TMA는 {lo}–{hi} Torr 범위여야 합니다 (학습 데이터 범위).")
    if cfg.P_total < cfg.P_HF + cfg.P_TMA + cfg.n2_min - 1e-9:
        msg.append(f"P_total ≥ P_HF + P_TMA + N₂ 최소 {cfg.n2_min} Torr 이어야 합니다 "
                   f"(현재 최소 {cfg.P_HF + cfg.P_TMA + cfg.n2_min:.2f} Torr 필요).")
    if cfg.P_total > O.N2_MAX + cfg.P_HF + cfg.P_TMA + 1e-9:
        msg.append(f"N₂ 배경압이 학습 범위({O.N2_MAX} Torr)를 넘습니다. P_total을 줄여 주세요.")
    return msg


def run(cfg, p=None, sur=None):
    """기준 / 입력 / 모델 A / 모델 B 결과와 제약 체크. optimize.main()과 같은 계산."""
    if p is None or sur is None:
        p, sur = load()
    base = O.physics(p, O.BASELINE["T"], O.BASELINE["P_HF"], O.BASELINE["P_TMA"], O.BASELINE["P_total"],
                     cfg.tau_scale, times=(O.BASELINE["t_HF"], O.BASELINE["tp_HF"], O.BASELINE["t_TMA"],
                                           O.BASELINE["tp_TMA"]))
    base.update(name="기준 레시피 (Lee 2016)", EPC_lit=O.BASELINE["EPC"])
    inp = O.physics(p, cfg.T, cfg.P_HF, cfg.P_TMA, cfg.P_total, cfg.tau_scale)
    c_ml, e_ml = sur(cfg.T, cfg.P_HF, cfg.P_TMA, cfg.P_total, cfg.tau_scale)
    inp.update(name="입력 레시피 (최소 시간)", EPC_ml=float(e_ml[0]), cycle_ml=float(c_ml[0]))
    A = O.model_A(p, sur, cfg)
    B = O.model_B(p, sur, cfg)
    out = {"base": base, "input": inp, "A": A, "B": B}
    for k in ("input", "A", "B"):
        if out[k] is not None:
            out[k]["checks"] = O.checks(out[k], cfg)
            # 마지막 항목(문헌 분압 범위)은 공정 제약이 아니라 외삽 경고라서 통과 여부와 분리한다
            out[k]["ok"] = all(c[1] for c in out[k]["checks"][:-1])
            out[k]["extrapolated"] = not out[k]["checks"][-1][1]
    return out


# ------------------------------------------------------------------ 물리식 기반 빠른 최적화
def physics_optimum(p, tau_scale, tmax, epc_min, loss=G.PURGE_LOSS, tmin=200.0, n2_min=G.N2_MIN):
    """모델 B와 같은 문제(총 cycle 최소, EPC ≥ 목표)를 물리 생성식만으로 푼다.
    대리모델은 허용 손실 5%로 학습돼 있어서, 허용 손실을 바꾸는 분석은 이 함수를 쓴다."""
    lo, hi = O.P_BOUNDS
    b = [(tmin, tmax), (np.log(lo), np.log(hi)), (np.log(lo), np.log(hi)), (np.log(n2_min), np.log(O.N2_MAX))]

    def recipe(x):
        T, PH, PT, PN = x[0], np.exp(x[1]), np.exp(x[2]), np.exp(x[3])
        return G.min_recipe(p, T, PH, PT, PN + PH + PT, ref="Z18", tau_scale=tau_scale, purge_loss=loss)

    def obj(x):
        x = np.atleast_2d(x.T).T
        r = recipe(x)
        return r["cycle_time"] + 1e3 * np.maximum(epc_min - r["EPC"], 0) - O.TIE_W * r["EPC"]

    res = O._de(obj, b)
    r = recipe(np.asarray(res.x).reshape(-1, 1))
    g = lambda k: float(np.ravel(r[k])[0])
    out = dict(T=g("T"), P_HF=g("P_HF"), P_TMA=g("P_TMA"), P_total=g("P_total"),
               t_react=g("t_react"), t_purge=g("t_purge"), cycle=g("cycle_time"), EPC=g("EPC"),
               tau_scale=tau_scale, loss=loss)
    out["feasible"] = out["EPC"] >= epc_min - 1e-6
    out["purge_share"] = out["t_purge"] / out["cycle"] if out["cycle"] > 0 else 0.0
    return out


def epc_ceiling(p, tmax):
    """그 T_max에서 포화 노출로 낼 수 있는 최대 EPC (분압과 거의 무관)"""
    return float(G.min_recipe(p, tmax, 0.3, 0.3, 1.5, ref="Z18")["EPC"])


def sweep_tmax(p, epc_min, tau_scale, loss=G.PURGE_LOSS, temps=range(250, 351, 10)):
    rows = []
    for tm in temps:
        row = dict(tmax=float(tm), EPC_max=epc_ceiling(p, tm))
        row["feasible"] = row["EPC_max"] >= epc_min
        if row["feasible"]:
            r = physics_optimum(p, tau_scale, float(tm), epc_min, loss)
            row.update(cycle=r["cycle"], T=r["T"], feasible=r["feasible"])
        rows.append(row)
    return rows


def sweep_tau(p, tmax, epc_min, loss=G.PURGE_LOSS, taus=(1, 3, 10, 30, 100)):
    return [physics_optimum(p, t, tmax, epc_min, loss) for t in taus]


def sweep_epc_target(p, tmax, tau_scale, loss=G.PURGE_LOSS, targets=(0.3, 0.5, 0.7, 0.9, 1.0, 1.1)):
    rows = []
    for e in targets:
        r = physics_optimum(p, tau_scale, tmax, e, loss)
        r["target"] = e
        rows.append(r)
    return rows


def sweep_loss(p, tmax, epc_min, losses=(0.01, 0.02, 0.05, 0.10), taus=(1, 10, 100)):
    return [physics_optimum(p, t, tmax, epc_min, l) for t in taus for l in losses]
