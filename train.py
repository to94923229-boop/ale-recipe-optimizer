"""
2단계-2: 대리모델 학습과 비교

입력 피처 : T, ln P_HF, ln P_TMA, ln P_total, ln tau_scale  (팀 공정 변수 4개 + 장비 설정)
목표      : log cycle_time,  EPC (학습은 EPC_noisy, 평가는 오차 없는 EPC)
파생      : throughput = EPC / cycle_time × 60  (두 모델의 예측으로 계산)

후보 모델 : Gaussian Process (800개 부분표본), HistGradientBoosting, RandomForest, MLP
평가      : 학습 80% 안에서 5-fold 교차검증 → 목표별 최적 모델 선택 → 시험 20%로 최종 평가
            문헌 분압 범위 안/밖(외삽)을 나눠서도 오차를 본다

실행:  python train.py
출력:  models/*.joblib, reports/train_cv.csv, reports/train_test.json, reports/train_report.png
"""
import sys
try:                                     # Windows 콘솔에서 Å, ≥, ✔ 같은 문자가 깨지지 않게
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "synthetic.csv"
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
REP = ROOT / "reports"
SEED = 0
GP_N = 800          # GP는 O(n³)이라 부분표본만 사용 (CPU 2개 환경 기준)


def features(df: pd.DataFrame) -> np.ndarray:
    """팀 공정 변수 4개(T, P_HF, P_TMA, P_total) + 장비 τ 배율. 압력·τ는 로그로 넣는다."""
    return np.column_stack([df["T"], np.log(df.P_HF), np.log(df.P_TMA), np.log(df.P_total), np.log(df.tau_scale)])


FEATURE_NAMES = ["T", "ln P_HF", "ln P_TMA", "ln P_total", "ln tau_scale"]
TARGETS = {"log_cycle": ("log cycle_time", None), "EPC": ("EPC (Å/cycle)", "EPC")}


def candidates():
    return {
        "GP": lambda: make_pipeline(StandardScaler(), GaussianProcessRegressor(
            ConstantKernel(1.0) * Matern(length_scale=np.ones(5), nu=2.5) + WhiteKernel(1e-3),
            normalize_y=True, n_restarts_optimizer=0, random_state=SEED)),
        "HistGB": lambda: HistGradientBoostingRegressor(max_iter=800, learning_rate=0.05, max_leaf_nodes=31,
                                                        l2_regularization=1e-3, random_state=SEED),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        "MLP": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 128, 64), alpha=1e-4,
                                     learning_rate_init=2e-3, max_iter=400, early_stopping=True,
                                     n_iter_no_change=30, random_state=SEED)),
    }


def target_values(df, key, clean=False):
    if key == "log_cycle":
        return np.log(df.cycle_time.values)
    return (df.EPC if clean else df.EPC_noisy).values


def cv_scores(train: pd.DataFrame):
    rows = []
    kf = KFold(5, shuffle=True, random_state=SEED)
    for name, make in candidates().items():
        sub = train.sample(GP_N, random_state=SEED) if name == "GP" else train
        X = features(sub)
        for key in TARGETS:
            y = target_values(sub, key); y_clean = target_values(sub, key, clean=True)
            preds = np.empty_like(y); t0 = time.time()
            for tr, va in kf.split(X):
                m = make(); m.fit(X[tr], y[tr]); preds[va] = m.predict(X[va])
            rows.append({"model": name, "target": key, "n_train": len(sub),
                         "R2": r2_score(y_clean, preds), "MAE": mean_absolute_error(y_clean, preds),
                         "sec": time.time() - t0})
            print(f"  {name:12s} {key:9s} R2={rows[-1]['R2']:.4f}  MAE={rows[-1]['MAE']:.4f}  ({rows[-1]['sec']:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA)
    train, test = train_test_split(df, test_size=0.2, random_state=SEED)
    print(f"train {len(train):,} / test {len(test):,}")
    print("5-fold 교차검증")
    cv = cv_scores(train)
    cv.to_csv(REP / "train_cv.csv", index=False)

    best, fitted = {}, {}
    for key in TARGETS:
        b = cv[cv.target == key].sort_values("MAE").iloc[0]
        best[key] = b.model
        sub = train.sample(GP_N, random_state=SEED) if b.model == "GP" else train
        m = candidates()[b.model](); m.fit(features(sub), target_values(sub, key))
        fitted[key] = m
        joblib.dump(m, MODELS / f"surrogate_{key}.joblib")

    Xt = features(test)
    p_cycle = np.exp(fitted["log_cycle"].predict(Xt))
    p_epc = fitted["EPC"].predict(Xt)
    p_thr = p_epc / p_cycle * 60
    test = test.assign(pred_cycle=p_cycle, pred_EPC=p_epc, pred_thr=p_thr)
    test.to_csv(REP / "train_test_predictions.csv", index=False)

    def metrics(g):
        etch = g.EPC > 0
        return {"n": int(len(g)),
                "cycle_time_MAPE_%": float(np.mean(np.abs(g.pred_cycle / g.cycle_time - 1)) * 100),
                "cycle_time_R2_log": float(r2_score(np.log(g.cycle_time), np.log(g.pred_cycle))),
                "EPC_MAE_A": float(mean_absolute_error(g.EPC, g.pred_EPC)),
                "EPC_R2": float(r2_score(g.EPC, g.pred_EPC)),
                "throughput_MAE_A_per_min(etch)": float(mean_absolute_error(g.throughput_A_per_min[etch], g.pred_thr[etch])),
                "throughput_R2(etch)": float(r2_score(g.throughput_A_per_min[etch], g.pred_thr[etch]))}

    res = {"best_model": best, "test_all": metrics(test),
           "test_in_lit_range": metrics(test[test.in_lit_range]),
           "test_out_of_lit_range": metrics(test[~test.in_lit_range]),
           "features": FEATURE_NAMES}
    (REP / "train_test.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    (MODELS / "surrogate_meta.json").write_text(json.dumps(
        {"features": FEATURE_NAMES, "best_model": best,
         "usage": "X = [T, ln P_HF, ln P_TMA, ln P_total, ln tau_scale]; cycle = exp(m_log_cycle.predict(X)); EPC = m_EPC.predict(X)"},
        indent=2, ensure_ascii=False))
    print("\n최적 모델:", best)
    for k in ("test_all", "test_in_lit_range", "test_out_of_lit_range"):
        print(k, {a: round(b, 4) for a, b in res[k].items()})
    report(cv, test, best)


def report(cv, test, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    import plotstyle
    plotstyle.setup()
    INK, SUB = "#17222D", "#566574"
    MC = {"GP": "#6D28D9", "HistGB": "#1F5FA8", "RandomForest": "#009B84", "MLP": "#D97706"}
    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(2, 3, top=0.85, bottom=0.08, left=0.08, right=0.98, hspace=0.5, wspace=0.38)
    fig.text(0.06, 0.94, "대리모델 학습 결과 (anchor 데이터 보정)", fontsize=15, weight="bold", color=INK)
    fig.text(0.06, 0.9, f"교차검증으로 고른 모델: cycle time → {best['log_cycle']}, EPC → {best['EPC']} · "
             "아래 줄은 시험셋(20%, 학습에 안 쓴 데이터) 결과", fontsize=9.5, color=SUB)
    for j, key in enumerate(TARGETS):
        a = fig.add_subplot(gs[0, j])
        c = cv[cv.target == key].sort_values("MAE")
        a.barh(c.model, c.MAE, color=[MC[m] for m in c.model], height=0.6)
        for y, (m, v) in enumerate(zip(c.model, c.MAE)):
            a.text(v, y, f"  {v:.4f}", va="center", fontsize=8.5, color=INK)
        a.invert_yaxis(); a.grid(axis="y", visible=False)
        a.set_title(f"교차검증 MAE — {'log cycle time' if key == 'log_cycle' else 'EPC (Å/cycle)'}",
                    loc="left", weight="bold", fontsize=10.5)
        a.set_xlim(0, c.MAE.max() * 1.35)
    a = fig.add_subplot(gs[0, 2]); a.axis("off")
    r = json.loads((REP / "train_test.json").read_text())
    lines = [("", "전체", "문헌 범위 안", "범위 밖"),
             ("cycle time 오차 (%)",) + tuple(f"{r[k]['cycle_time_MAPE_%']:.2f}" for k in ("test_all", "test_in_lit_range", "test_out_of_lit_range")),
             ("EPC MAE (Å)",) + tuple(f"{r[k]['EPC_MAE_A']:.4f}" for k in ("test_all", "test_in_lit_range", "test_out_of_lit_range")),
             ("throughput MAE (Å/min)",) + tuple(f"{r[k]['throughput_MAE_A_per_min(etch)']:.3f}" for k in ("test_all", "test_in_lit_range", "test_out_of_lit_range")),
             ("throughput R²",) + tuple(f"{r[k]['throughput_R2(etch)']:.4f}" for k in ("test_all", "test_in_lit_range", "test_out_of_lit_range"))]
    t = a.table(cellText=lines[1:], colLabels=lines[0], loc="center", cellLoc="center", colWidths=[0.4, 0.2, 0.22, 0.18])
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.6)
    for (i, jj), cell in t.get_celld().items():
        cell.set_edgecolor("#D5DDE3")
        if i == 0: cell.set_text_props(weight="bold")
    a.set_title("시험셋 성능", loc="left", weight="bold", fontsize=10.5)
    k = dict(s=4, alpha=0.3, lw=0)
    a = fig.add_subplot(gs[1, 0])
    a.scatter(test.cycle_time, test.pred_cycle, color=MC[best["log_cycle"]], **k)
    lim = [test.cycle_time.min() * 0.8, test.cycle_time.max() * 1.2]
    a.plot(lim, lim, color=SUB, lw=1); a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("생성식 cycle time (s)"); a.set_ylabel("대리모델 예측 (s)"); a.set_title("cycle time", loc="left", weight="bold")
    a = fig.add_subplot(gs[1, 1])
    a.scatter(test.EPC, test.pred_EPC, color=MC[best["EPC"]], **k)
    lim = [test.EPC.min() - 0.05, test.EPC.max() + 0.05]
    a.plot(lim, lim, color=SUB, lw=1)
    a.set_xlabel("생성식 EPC (Å/cycle)"); a.set_ylabel("대리모델 예측"); a.set_title("EPC", loc="left", weight="bold")
    a = fig.add_subplot(gs[1, 2])
    e = test[test.EPC > 0]
    a.scatter(e.throughput_A_per_min, e.pred_thr, color=INK, **k)
    lim = [0, e.throughput_A_per_min.max() * 1.05]
    a.plot(lim, lim, color=SUB, lw=1)
    a.set_xlabel("생성식 throughput (Å/min)"); a.set_ylabel("예측 (EPC / cycle time)")
    a.set_title("throughput (식각 조건, 파생값)", loc="left", weight="bold")
    fig.savefig(REP / "train_report.png", dpi=150)


if __name__ == "__main__":
    main()
