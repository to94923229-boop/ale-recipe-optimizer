"""
3단계: 대리모델(ML)로 최적 레시피 찾기 — 모델 A / 모델 B  (팀 최종 목표)

목표: 공정 품질(EPC, 반응 포화, 잔류 전구체 제거)을 유지하면서 전체 cycle time을 최소화하는 공정 조건

입력 (웹사이트 INPUT과 같음): T, P_HF, P_TMA, P_total  — 기본값은 기준 레시피(Lee 2016)
          장비 설정: tau_scale (퍼지 τ 배율, Zywotko 보정 장비 = 1)

  모델 A (반응시간 최적화)  변수 T, P_HF, P_TMA  · P_total 고정(입력값) · 퍼지 고정(가장 높은 분압에서도 충분한 값)
                           목표: HF + TMA 주입시간 합 최소 · 제약: T ≤ T_max, 분압 범위, EPC ≥ 목표
  모델 B (전체 cycle 최적화) 변수 T, P_HF, P_TMA, P_total · 퍼지는 조건별 필요한 만큼(퍼지 모델)
                           목표: 총 cycle time 최소 · 제약: 포화, 잔류 손실 ≤ 5%, P_N2 = P_total − P_HF − P_TMA ≥ 0.5 Torr,
                           EPC ≥ 목표

방법: ML 예측 + 물리 기반 최적화
  - 대리모델(models/*.joblib)로 cycle time·EPC를 예측하며 differential evolution으로 탐색
  - 찾은 조건을 물리 생성식(generator.min_recipe)으로 다시 계산해 권장 주입·퍼지 시간과 제약 충족을 확인

실행:  python optimize.py                      (기본: 기준 레시피를 입력으로)
       python optimize.py --T 300 --P_HF 0.2 --P_TMA 0.1 --P_total 1.5 --tmax 320 --epc-min 0.9 --tau-scale 30
출력:  reports/optimization_report.pdf, reports/opt_*.png, reports/optimization_summary.csv/.json
       터미널 맨 아래에 입력 레시피·모델 A·모델 B 결과 요약
"""
from __future__ import annotations

import sys
try:                                     # Windows 콘솔에서 Å, ≥, ✔ 같은 문자가 깨지지 않게
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import differential_evolution
import generator as G
import train as TR

ROOT = Path(__file__).parent
REP = ROOT / "reports"
MODELS = ROOT / "models"

# 기준 레시피: Lee 2016 (팀 참고 논문 1번) 실험 조건과 측정값
BASELINE = dict(name="기준 레시피 (Lee 2016 문헌 조건)", T=300.0, P_HF=0.08, P_TMA=0.04, P_total=1.12,
                t_HF=1.0, tp_HF=30.0, t_TMA=2.0, tp_TMA=30.0, EPC=0.46)
P_BOUNDS = (0.02, 1.0)        # 분압 탐색 범위 (Torr) = 학습 데이터 범위
N2_MAX = 2.5                  # 배경 N2 상한 (Torr) = 학습 데이터 범위
LIT_P = (0.04, 8.0)           # 문헌이 다룬 분압 범위
TMA_PVAP = 7.0                # TMA 버블러 증기압 상한 (Torr), 장비에 맞게 수정
SEED = 0
ML_MARGIN = 0.015             # ML로 탐색할 때 EPC 목표에 더하는 여유 (대리모델 시험 오차의 약 3배)
TIE_W = 0.02                  # 같은 시간이면 EPC(→ throughput)가 높은 쪽을 고르도록 하는 작은 가중치 (s per Å)


def _de(obj, bounds):
    """대리모델을 한 번에 여러 점씩 평가하는 differential evolution (재현 가능하도록 seed 고정)"""
    return differential_evolution(obj, bounds, seed=SEED, tol=1e-7, maxiter=150, popsize=15, polish=False,
                                  vectorized=True, updating="deferred")


# ------------------------------------------------------------------ 대리모델
class Surrogate:
    def __init__(self):
        self.m_cyc = joblib.load(MODELS / "surrogate_log_cycle.joblib")
        self.m_epc = joblib.load(MODELS / "surrogate_EPC.joblib")

    def __call__(self, T, P_HF, P_TMA, P_total, tau_scale):
        T, P_HF, P_TMA, P_total, tau_scale = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, float))
                                                                  for v in (T, P_HF, P_TMA, P_total, tau_scale)))
        df = pd.DataFrame(dict(T=T, P_HF=P_HF, P_TMA=P_TMA, P_total=P_total, tau_scale=tau_scale))
        X = TR.features(df)
        return np.exp(self.m_cyc.predict(X)), self.m_epc.predict(X)


# ------------------------------------------------------------------ 물리 생성식으로 레시피 계산
def physics(p, T, P_HF, P_TMA, P_total, tau_scale, tp_fixed=None, times=None):
    """권장 주입·퍼지 시간, EPC, 포화도, 잔류 손실.
    tp_fixed=(tp_HF, tp_TMA): 퍼지를 고정값으로 (모델 A) · times=(t_HF, tp_HF, t_TMA, tp_TMA): 주어진 레시피 그대로"""
    r = G.min_recipe(p, T, P_HF, P_TMA, P_total, ref="Z18", tau_scale=tau_scale)
    tau, tauHF = float(r["tau"]), float(r["tau_HF"])
    t_HF, t_TMA = float(r["t_HF"]), float(r["t_TMA"])
    tp_HF, tp_TMA = float(r["tp_HF"]), float(r["tp_TMA"])
    if tp_fixed is not None:
        tp_HF, tp_TMA = tp_fixed
    if times is not None:
        t_HF, tp_HF, t_TMA, tp_TMA = times
    X0h, X0t = np.exp(p["lnX0_HF"]), np.exp(p["lnX0_TMA"])
    epc = float(G._epc_ref(p, T, P_HF, P_TMA, t_HF, t_TMA, tp_HF, tp_TMA, tau, tauHF)["EPC"])
    L, _ = G._loss(p, P_HF / X0h, P_TMA / X0t, tp_HF, tp_TMA, tau, tauHF)
    cycle = t_HF + tp_HF + t_TMA + tp_TMA
    return dict(T=float(T), P_HF=float(P_HF), P_TMA=float(P_TMA), P_total=float(P_total),
                P_N2=float(P_total - P_HF - P_TMA),
                t_HF=t_HF, tp_HF=tp_HF, t_TMA=t_TMA, tp_TMA=tp_TMA,
                t_react=t_HF + t_TMA, t_purge=tp_HF + tp_TMA, cycle=cycle, EPC=epc,
                throughput=epc / cycle * 60.0,
                sat_HF=float(1 - np.exp(-P_HF * t_HF / X0h)), sat_TMA=float(1 - np.exp(-P_TMA * t_TMA / X0t)),
                purge_loss=float(1 - L), tau=tau, tau_HF=tauHF)


def checks(r, cfg):
    """공정 제약 충족 여부 (이름, 통과?, 설명)"""
    return [
        ("EPC ≥ 목표", r["EPC"] >= cfg.epc_min - 1e-9, f"{r['EPC']:.3f} ≥ {cfg.epc_min:.2f} Å/cycle"),
        ("T ≤ T_max", r["T"] <= cfg.tmax + 1e-6, f"{r['T']:.0f} ≤ {cfg.tmax:.0f} °C"),
        ("HF 반응 포화", r["sat_HF"] >= G.SAT_LEVEL - 1e-3, f"{r['sat_HF']:.0%} (기준 {G.SAT_LEVEL:.0%})"),
        ("TMA 반응 포화", r["sat_TMA"] >= G.SAT_LEVEL - 1e-3, f"{r['sat_TMA']:.0%} (기준 {G.SAT_LEVEL:.0%})"),
        ("잔류 전구체 제거", r["purge_loss"] <= G.PURGE_LOSS + 1e-3,
         f"잔류에 의한 EPC 손실 {r['purge_loss']:.1%} (허용 {G.PURGE_LOSS:.0%})"),
        ("N2 배경압 확보", r["P_N2"] >= cfg.n2_min - 1e-6, f"P_N2 = {r['P_N2']:.2f} ≥ {cfg.n2_min:.2f} Torr"),
        ("TMA 증기압 이하", r["P_TMA"] <= TMA_PVAP, f"{r['P_TMA']:.2f} ≤ {TMA_PVAP:.0f} Torr"),
        ("문헌 분압 범위 안", LIT_P[0] <= min(r["P_HF"], r["P_TMA"]) and max(r["P_HF"], r["P_TMA"]) <= LIT_P[1],
         f"{LIT_P[0]}–{LIT_P[1]:g} Torr (밖이면 외삽)"),
    ]


# ------------------------------------------------------------------ 모델 A / B
def fixed_purge(p, cfg, P_total):
    """모델 A의 고정 퍼지: P_total 고정에서 가장 높은 분압(=가장 많은 잔류)에도 충분한 퍼지"""
    smax = P_total - cfg.n2_min
    g = np.geomspace(P_BOUNDS[0], max(smax - P_BOUNDS[0], P_BOUNDS[0] * 1.01), 25)
    PH, PT = np.meshgrid(g, g)
    ok = PH + PT <= smax + 1e-9
    r = G.min_recipe(p, cfg.tmax, PH[ok], PT[ok], P_total, ref="Z18", tau_scale=cfg.tau_scale)
    return float(np.max(r["tp_HF"])), float(np.max(r["tp_TMA"]))


def model_A(p, sur, cfg, P_total=None):
    P_total = cfg.P_total if P_total is None else P_total
    smax = P_total - cfg.n2_min
    if smax < 2 * P_BOUNDS[0]:
        return None
    X0h, X0t = np.exp(p["lnX0_HF"]), np.exp(p["lnX0_TMA"])
    nsat = -np.log(1 - G.SAT_LEVEL)
    tp_fix = fixed_purge(p, cfg, P_total)
    hi = np.log(min(P_BOUNDS[1], smax))

    for margin in (ML_MARGIN, 2 * ML_MARGIN, 4 * ML_MARGIN):
        def obj(x):
            x = np.atleast_2d(x.T).T
            T, PH, PT = x[0], np.exp(x[1]), np.exp(x[2])
            react = nsat * (X0h / PH + X0t / PT)                   # 물리: 포화에 필요한 주입시간 합
            _, epc = sur(T, PH, PT, np.full_like(T, P_total), cfg.tau_scale)   # ML: EPC 예측
            pen = 1e3 * np.maximum(cfg.epc_min + margin - epc, 0) + 1e3 * np.maximum(PH + PT - smax, 0)
            return react + pen - TIE_W * epc

        res = _de(obj, [(cfg.tmin, cfg.tmax), (np.log(P_BOUNDS[0]), hi), (np.log(P_BOUNDS[0]), hi)])
        T, PH, PT = res.x[0], np.exp(res.x[1]), np.exp(res.x[2])
        r = physics(p, T, PH, PT, P_total, cfg.tau_scale, tp_fixed=tp_fix)
        if r["EPC"] >= cfg.epc_min:                                # 물리식 검증 통과하면 종료
            break
    cyc_ml, epc_ml = sur(T, PH, PT, P_total, cfg.tau_scale)
    r.update(name="모델 A (반응시간 최적화)", feasible=r["EPC"] >= cfg.epc_min, EPC_ml=float(epc_ml[0]),
             cycle_ml_minrecipe=float(cyc_ml[0]),
             objective="HF + TMA 주입시간 합", fixed_purge=tp_fix)
    return r


def model_B(p, sur, cfg, tau_scale=None, epc_min=None, tmax=None):
    ts = cfg.tau_scale if tau_scale is None else tau_scale
    emin = cfg.epc_min if epc_min is None else epc_min
    tm = cfg.tmax if tmax is None else tmax

    b = [(cfg.tmin, tm), (np.log(P_BOUNDS[0]), np.log(P_BOUNDS[1])), (np.log(P_BOUNDS[0]), np.log(P_BOUNDS[1])),
         (np.log(cfg.n2_min), np.log(N2_MAX))]
    for margin in (ML_MARGIN, 2 * ML_MARGIN, 4 * ML_MARGIN):
        def obj(x):
            x = np.atleast_2d(x.T).T
            T, PH, PT, PN = x[0], np.exp(x[1]), np.exp(x[2]), np.exp(x[3])
            cyc, epc = sur(T, PH, PT, PN + PH + PT, np.full_like(T, ts))
            return cyc + 1e3 * np.maximum(emin + margin - epc, 0) - TIE_W * epc

        res = _de(obj, b)
        T, PH, PT, PN = res.x[0], np.exp(res.x[1]), np.exp(res.x[2]), np.exp(res.x[3])
        Pt = PN + PH + PT
        r = physics(p, T, PH, PT, Pt, ts)
        if r["EPC"] >= emin:                                       # 물리식 검증 통과하면 종료
            break
    cyc_ml, epc_ml = sur(T, PH, PT, Pt, ts)
    r.update(name="모델 B (전체 cycle 최적화)", feasible=r["EPC"] >= emin, EPC_ml=float(epc_ml[0]), cycle_ml=float(cyc_ml[0]),
             objective="총 cycle time", tau_scale=ts)
    return r


# ------------------------------------------------------------------ 보조 분석 (trade-off)
def tradeoff_Ptotal(p, cfg, PH, PT):
    """모델 B 최적 분압에서 P_total만 바꿀 때 반응시간·퍼지·cycle"""
    Pt = np.linspace(PH + PT + cfg.n2_min, PH + PT + N2_MAX, 50)
    r = G.min_recipe(p, cfg.tmax, PH, PT, Pt, ref="Z18", tau_scale=cfg.tau_scale)
    return Pt, r


def tradeoff_scale(p, cfg, best):
    """분압을 같은 비율로 키울 때: 주입시간 ↓, 퍼지 ↑ (P_N2는 모델 B 최적값 고정)"""
    k = np.geomspace(0.1, 1.0, 40) * min(P_BOUNDS[1] / max(best["P_HF"], best["P_TMA"]), 20)
    k = np.unique(np.r_[k, 1.0])
    PH, PT = best["P_HF"] * k, best["P_TMA"] * k
    ok = (PH >= P_BOUNDS[0]) & (PT >= P_BOUNDS[0]) & (PH <= P_BOUNDS[1]) & (PT <= P_BOUNDS[1])
    PH, PT = PH[ok], PT[ok]
    r = G.min_recipe(p, best["T"], PH, PT, best["P_N2"] + PH + PT, ref="Z18", tau_scale=best.get("tau_scale", cfg.tau_scale))
    return PH + PT, r


def main():
    ap = argparse.ArgumentParser(description="모델 A/B 최적 레시피")
    ap.add_argument("--T", type=float, default=BASELINE["T"], help="입력 레시피 온도 (°C)")
    ap.add_argument("--P_HF", type=float, default=BASELINE["P_HF"], help="입력 HF 분압 (Torr)")
    ap.add_argument("--P_TMA", type=float, default=BASELINE["P_TMA"], help="입력 TMA 분압 (Torr)")
    ap.add_argument("--P_total", type=float, default=BASELINE["P_total"], help="입력 총압 (Torr) — 모델 A에서 고정")
    ap.add_argument("--tmin", type=float, default=200.0)
    ap.add_argument("--tmax", type=float, default=300.0, help="허용 최고 온도 T_max (°C)")
    ap.add_argument("--epc-min", type=float, default=0.9, help="EPC 목표 하한 (Å/cycle)")
    ap.add_argument("--n2-min", type=float, default=G.N2_MIN, help="최소 N2 배경압 (Torr)")
    ap.add_argument("--tau-scale", type=float, default=1.0, help="장비 퍼지 τ 배율 (Zywotko 보정 장비 = 1)")
    ap.add_argument("--no-report", action="store_true", help="PDF 리포트 생략 (빠른 실행)")
    cfg = ap.parse_args()

    p = G.load_params(); sur = Surrogate()
    # 기준 레시피 (문헌 조건 그대로, 측정 EPC) 와 입력 레시피 (같은 압력에서 최소 시간으로 바꿨을 때)
    base = physics(p, BASELINE["T"], BASELINE["P_HF"], BASELINE["P_TMA"], BASELINE["P_total"], cfg.tau_scale,
                   times=(BASELINE["t_HF"], BASELINE["tp_HF"], BASELINE["t_TMA"], BASELINE["tp_TMA"]))
    base.update(name=BASELINE["name"], EPC_lit=BASELINE["EPC"])
    inp = physics(p, cfg.T, cfg.P_HF, cfg.P_TMA, cfg.P_total, cfg.tau_scale)
    c_ml, e_ml = sur(cfg.T, cfg.P_HF, cfg.P_TMA, cfg.P_total, cfg.tau_scale)
    inp.update(name="입력 레시피 (같은 조건, 최소 시간)", EPC_ml=float(e_ml[0]), cycle_ml=float(c_ml[0]))
    print("모델 A 최적화 중..."); A = model_A(p, sur, cfg)
    print("모델 B 최적화 중..."); B = model_B(p, sur, cfg)
    results = [base, inp] + ([A] if A else []) + [B]

    extra = {}
    if not cfg.no_report:
        print("보조 분석 (T_max 스윕, 장비 τ 민감도)...")
        extra["tmax_sweep"] = []
        for tm in np.arange(250, 351, 10.0):
            # 포화 노출에서 그 온도로 낼 수 있는 최대 EPC (분압과 거의 무관) → 목표를 넘어야 최적화 의미가 있음
            emax = float(G.min_recipe(p, tm, 0.3, 0.3, 1.5, ref="Z18")["EPC"])
            row = {"tmax": tm, "EPC_max": emax, "feasible": emax >= cfg.epc_min,
                   "A_react": np.nan, "A_cycle": np.nan, "B_cycle": np.nan, "B_T": np.nan}
            if row["feasible"]:
                c2 = argparse.Namespace(**{**vars(cfg), "tmax": tm})
                a = model_A(p, sur, c2); b = model_B(p, sur, c2, tmax=tm)
                row.update(A_react=a["t_react"] if a else np.nan, A_cycle=a["cycle"] if a else np.nan,
                           B_cycle=b["cycle"], B_T=b["T"])
            extra["tmax_sweep"].append(row)
        extra["tau_sweep"] = [model_B(p, sur, cfg, tau_scale=ts) for ts in (1, 3, 10, 30, 100)]
        T = np.linspace(200, 350, 151)
        extra["epc_T"] = (T, G.min_recipe(p, T, 0.3, 0.3, 1.5, ref="Z18")["EPC"])
        import opt_report
        opt_report.make(results, extra, cfg, p)

    save(results, extra, cfg)
    print_summary(results, cfg)


def save(results, extra, cfg):
    keys = ["name", "T", "P_HF", "P_TMA", "P_total", "P_N2", "t_HF", "tp_HF", "t_TMA", "tp_TMA", "t_react", "t_purge",
            "cycle", "EPC", "throughput", "sat_HF", "sat_TMA", "purge_loss"]
    pd.DataFrame([{k: r.get(k) for k in keys} for r in results]).to_csv(REP / "optimization_summary.csv", index=False,
                                                                     encoding="utf-8-sig")
    out = {"settings": vars(cfg), "results": [{k: (v if not isinstance(v, (np.floating,)) else float(v)) for k, v in r.items()}
                                              for r in results],
           "checks": {r["name"]: [(n, bool(ok), s) for n, ok, s in checks(r, cfg)] for r in results}}
    if extra:
        out["tmax_sweep"] = extra["tmax_sweep"]
        out["tau_sweep"] = [{k: r[k] for k in ("tau_scale", "T", "P_HF", "P_TMA", "P_total", "t_react", "t_purge", "cycle", "EPC")}
                            for r in extra["tau_sweep"]]
    (REP / "optimization_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))


def print_summary(results, cfg):
    base = results[0]
    line = "=" * 78
    print("\n" + line)
    print(" Al2O3 HF/TMA Thermal ALE — 최적 레시피 요약")
    print(f" 설정: T_max {cfg.tmax:.0f}°C · EPC 목표 ≥ {cfg.epc_min:.2f} Å/cycle · 장비 τ 배율 {cfg.tau_scale:g}"
          f" · 최소 N2 {cfg.n2_min:.2f} Torr · 포화 {G.SAT_LEVEL:.0%} · 잔류 손실 ≤ {G.PURGE_LOSS:.0%}")
    print(f" 기준: {base['name']} — cycle {base['cycle']:.1f} s (주입 {base['t_react']:.1f} + 퍼지 {base['t_purge']:.1f}),"
          f" EPC {base['EPC_lit']:.2f} Å/cycle(측정), {base['EPC_lit'] / base['cycle'] * 60:.2f} Å/min")
    print(line)
    for r in results[1:]:
        print(f"\n■ {r['name']}")
        print(f"  [입력 변수]  T {r['T']:.1f} °C | P_HF {r['P_HF']:.3f} | P_TMA {r['P_TMA']:.3f} | "
              f"P_total {r['P_total']:.3f} Torr (N2 {r['P_N2']:.2f})")
        if "fixed_purge" in r:
            print(f"  [고정 조건]  P_total 고정, 퍼지 고정 = HF후 {r['fixed_purge'][0]:.2f} s / TMA후 {r['fixed_purge'][1]:.2f} s")
        print(f"  ① 권장 HF/TMA Exposure Time : HF {r['t_HF']:.2f} s, TMA {r['t_TMA']:.2f} s   (반응 합 {r['t_react']:.2f} s)")
        print(f"  ② 권장 Purge Time           : HF후 {r['tp_HF']:.2f} s, TMA후 {r['tp_TMA']:.2f} s   (퍼지 합 {r['t_purge']:.2f} s)")
        print(f"  ③ 최소 총 Cycle Time        : {r['cycle']:.2f} s   (기준 {base['cycle']:.0f} s 대비 "
              f"{(1 - r['cycle'] / base['cycle']):.1%} 감소)")
        ml = r.get("EPC_ml")
        print(f"  ④ EPC                       : {r['EPC']:.3f} Å/cycle (물리식)"
              + (f" / {ml:.3f} (ML 예측)" if ml is not None else "") + f"   → throughput {r['throughput']:.2f} Å/min")
        print("     공정 제약 충족 여부:")
        for n, ok, s in checks(r, cfg):
            print(f"       {'✔' if ok else '✘'} {n:<10s} {s}")
        share = r["t_purge"] / r["cycle"] if r["cycle"] > 0 else 0
        print(f"  ⑤ 반응시간–Purge Trade-off   : 반응 {r['t_react']:.2f} s ({1 - share:.0%}) vs 퍼지 {r['t_purge']:.2f} s ({share:.0%})"
              f" · 퍼지 감소 {base['t_purge']:.0f} → {r['t_purge']:.2f} s")
        ok_all = all(ok for n, ok, _ in checks(r, cfg) if n != "문헌 분압 범위 안")
        if r.get("feasible") is False:
            print(f"  ▶ 최종 목표: 미달 — T_max {cfg.tmax:.0f}°C에서는 EPC 목표 {cfg.epc_min:.2f}를 낼 수 없음 "
                  "(포화 노출에서 EPC는 온도로 정해짐. T_max를 올리거나 EPC 목표를 낮출 것)")
        else:
            print(f"  ▶ 최종 목표(품질 유지 + cycle 최소화): {'달성' if ok_all else '미달 — 위 ✘ 항목 확인'}")
    print("\n" + line)
    print(" 리포트: reports/optimization_report.pdf · 표: reports/optimization_summary.csv")
    print(line)


if __name__ == "__main__":
    main()
