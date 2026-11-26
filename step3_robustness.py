"""
Step 3: robustness test. How do accuracy and the score change when the
HELD-OUT audio is degraded?

RESEARCH PROTOTYPE, NOT A DIAGNOSIS.

Procedure:
  1. For every recording and every condition in audio_tools.CONDITIONS (added
     noise at 30/20/10/5/0 dB SNR, MP3 at 32/16/8 kbps, 4 kHz resample), degrade
     the audio and re-run the SAME extract_features() used for training.
  2. Grouped 5-fold CV, repeated 5 times: train on CLEAN features of the training
     speakers, then score the test speakers' CLEAN and DEGRADED versions.
     So the model never sees a test speaker, and never sees degraded audio in training.
  3. Report accuracy, sensitivity, specificity, AUC, the mean P(Parkinson's)
     per group, and how far each recording's score moved from its clean score.

Outputs: results/step3_degraded_features.csv, results/step3_robustness.csv,
         results/step3_feature_drift.csv, results/step3_robustness.png
Run:     .venv/bin/python step3_robustness.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import parselmouth
from sklearn.metrics import roc_auc_score

from features import extract_features, FEATURE_COLUMNS
from audio_tools import CONDITIONS
from common import (make_model, grouped_folds, style_axes, stamp_prototype,
                    HC_COLOR, PD_COLOR, TEXT, TEXT_2, plt)

OUT = Path("results")
N_REPEATS_ROBUST = 5      # 5 repeats x 5 folds = 25 test folds per condition


def degraded_feature_table(meta: pd.DataFrame) -> pd.DataFrame:
    """Re-extract features for every (recording, condition). Cached to CSV (slow step)."""
    cache = OUT / "step3_degraded_features.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    for i, r in meta.iterrows():
        snd = parselmouth.Sound(str(r.path))
        for name, family, level, fn in CONDITIONS:
            # seed depends on the recording, so each file gets its own noise sample
            feats = extract_features(fn(snd, 1000 + i))
            rows.append({"sample_id": r.sample_id, "label": r.label,
                         "condition": name, **feats})
        print(f"[{i + 1:2d}/{len(meta)}] degraded + re-extracted {r.sample_id[:16]}")
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def main():
    from step1_extract_features import load_metadata
    meta = load_metadata()
    deg = degraded_feature_table(meta)
    # Praat can fail to find a pitch in very noisy audio; fill any gap with the
    # training mean so the model still gives an answer (counted below).
    n_missing = deg[FEATURE_COLUMNS].isna().any(axis=1).groupby(deg.condition).sum()

    clean = deg[deg.condition == "clean"].set_index("sample_id").loc[meta.sample_id]
    y, groups = clean["label"].to_numpy(), clean.index.to_numpy()
    Xc = clean[FEATURE_COLUMNS].to_numpy(float)
    by_cond = {c: deg[deg.condition == c].set_index("sample_id").loc[groups, FEATURE_COLUMNS]
               for c, *_ in CONDITIONS}

    rows, scores = [], []
    for model_name in ("logreg", "rf"):
        for rep in range(N_REPEATS_ROBUST):
            for fold, (train, test) in enumerate(grouped_folds(y, groups, rep)):
                model = make_model(model_name).fit(Xc[train], y[train])
                train_mean = Xc[train].mean(axis=0)
                p_clean = model.predict_proba(Xc[test])[:, 1]
                for cond, family, level, _ in CONDITIONS:
                    Xt = by_cond[cond].iloc[test].to_numpy(float)
                    Xt = np.where(np.isnan(Xt), train_mean, Xt)
                    p = model.predict_proba(Xt)[:, 1]
                    pred, truth = (p >= 0.5).astype(int), y[test]
                    rows.append({"model": model_name, "repeat": rep, "fold": fold,
                                 "condition": cond, "family": family, "level": level,
                                 "acc": np.mean(pred == truth),
                                 "sens": np.mean(pred[truth == 1] == 1),
                                 "spec": np.mean(pred[truth == 0] == 0)})
                    scores += [{"model": model_name, "repeat": rep, "condition": cond,
                                "sample_id": s, "label": t, "p_pd": pp, "p_clean": pc}
                               for s, t, pp, pc in zip(groups[test], truth, p, p_clean)]
    folds, scores = pd.DataFrame(rows), pd.DataFrame(scores)

    # ---- Summary table -------------------------------------------------------
    order = [c for c, *_ in CONDITIONS]
    summ = folds.groupby(["model", "condition"]).agg(
        acc_mean=("acc", "mean"), acc_sd=("acc", "std"),
        sens_mean=("sens", "mean"), spec_mean=("spec", "mean")).reset_index()
    scores["abs_change"] = (scores.p_pd - scores.p_clean).abs()
    s2 = scores.groupby(["model", "condition"]).apply(lambda d: pd.Series({
        "auc": np.mean([roc_auc_score(g.label, g.p_pd) for _, g in d.groupby("repeat")]),
        "mean_p_pd_HC": d[d.label == 0].p_pd.mean(),
        "mean_p_pd_PD": d[d.label == 1].p_pd.mean(),
        "mean_abs_score_change": d.abs_change.mean(),
        "pct_flipped": np.mean((d.p_pd >= 0.5) != (d.p_clean >= 0.5)) * 100,
    }), include_groups=False).reset_index()
    summ = summ.merge(s2, on=["model", "condition"])
    summ["files_with_failed_measurement"] = summ.condition.map(n_missing).fillna(0).astype(int)
    summ["condition"] = pd.Categorical(summ.condition, order, ordered=True)
    summ = summ.sort_values(["model", "condition"])
    summ.to_csv(OUT / "step3_robustness.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))

    # ---- Feature drift: how far does each feature move, in training-SD units? ------
    sd = clean[FEATURE_COLUMNS].std()
    drift = pd.DataFrame({c: ((by_cond[c] - clean[FEATURE_COLUMNS]) / sd).mean()
                          for c in order[1:]}).T
    drift.to_csv(OUT / "step3_feature_drift.csv")
    print("\nMean feature shift vs clean (in training standard deviations):")
    print(drift[["f0_mean_hz", "jitter_local", "shimmer_local", "hnr_db", "mfcc0_mean",
                 "mfcc1_mean"]].round(2).to_string())

    plot(summ, folds)


def plot(summ, folds):
    """Panel A: accuracy per condition. Panel B: mean P(PD) for each true group (logreg)."""
    order = [c for c, *_ in CONDITIONS]
    labels = ["Clean"] + [f"{fam}\n{lvl}" for _, fam, lvl, _ in CONDITIONS[1:]]
    x = np.arange(len(order))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)

    for k, (model, color, name) in enumerate([("logreg", HC_COLOR, "Logistic regression"),
                                              ("rf", PD_COLOR, "Random forest")]):
        f = folds[folds.model == model]
        m = f.groupby("condition").acc.mean().loc[order]
        lo = f.groupby("condition").acc.quantile(0.1).loc[order]
        hi = f.groupby("condition").acc.quantile(0.9).loc[order]
        xx = x + (k - 0.5) * 0.18
        a1.errorbar(xx, m, yerr=[m - lo, hi - m], fmt="o", ms=7, color=color,
                    ecolor=color, elinewidth=1.2, capsize=0, label=name)
    a1.axhline(0.5, color=TEXT, ls="--", lw=1)
    a1.text(-0.45, 0.505, "chance", ha="left", va="bottom", fontsize=8, color=TEXT)
    a1.set_ylim(0.2, 1.0)
    a1.set_ylabel("Accuracy on held-out speakers\n(dot = mean, bar = 10th-90th pct of folds)")
    a1.legend(loc="upper right", frameon=False, fontsize=8)
    a1.set_title("A. Accuracy when the test audio is degraded (model trained on clean audio)",
                 loc="left", fontsize=10, color=TEXT)

    s = summ[summ.model == "logreg"].set_index("condition").loc[order]
    # Connect points only WITHIN a degradation family (never across two kinds of
    # damage). Faint dotted lines mark each group's clean-audio score for reference.
    for col, color in (("mean_p_pd_PD", PD_COLOR), ("mean_p_pd_HC", HC_COLOR)):
        a2.axhline(s[col].iloc[0], color=color, ls=":", lw=1, alpha=0.7)
        fam_of = [fam for _, fam, _, _ in CONDITIONS]
        for fam in dict.fromkeys(fam_of):
            idx = [i for i, f in enumerate(fam_of) if f == fam]
            if len(idx) > 1:
                a2.plot(x[idx], s[col].iloc[idx], "-", color=color, lw=2)
    a2.plot(x, s.mean_p_pd_PD, "o", color=PD_COLOR, ms=7, label="True Parkinson's speakers")
    a2.plot(x, s.mean_p_pd_HC, "o", color=HC_COLOR, ms=7, label="True healthy controls")
    a2.text(x[-1] + 0.15, s.mean_p_pd_PD.iloc[-1], "Parkinson's", color=TEXT, fontsize=8, va="center")
    a2.text(x[-1] + 0.15, s.mean_p_pd_HC.iloc[-1], "Healthy", color=TEXT, fontsize=8, va="center")
    a2.axhline(0.5, color=TEXT, ls="--", lw=1)
    a2.set_ylim(0, 1)
    a2.set_ylabel("Mean model score\nP(Parkinson's), logistic regression")
    a2.legend(loc="upper left", frameon=False, fontsize=8)
    a2.set_title("B. Mean score by true group (dotted = clean level). Both groups moving together = the degradation, not the voice, drives the score",
                 loc="left", fontsize=10, color=TEXT)
    a2.set_xticks(x, labels, fontsize=8)
    a2.set_xlim(-0.5, len(order) - 0.1)
    for ax in (a1, a2):
        style_axes(ax)
        for b in (0.5, 5.5, 8.5):          # separate the degradation families
            ax.axvline(b, color="#e6e6e6", lw=1)
    stamp_prototype(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "step3_robustness.png", dpi=150)


if __name__ == "__main__":
    main()
