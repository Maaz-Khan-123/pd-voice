"""
Step 2: baseline classifiers with grouped cross-validation, plus sanity checks.

RESEARCH PROTOTYPE, NOT A DIAGNOSIS.

Questions answered:
  A. How well do VOICE features separate Parkinson's from healthy controls?
  B. SANITY CHECK: how well do RECORDING features alone (duration, loudness,
     noise floor, leading/trailing silence) do? If they work as well as voice,
     the voice result could be a recording-setup artefact.
  C. Do AGE or SEX alone predict the label? (Age is heavily confounded here.)
  D. Does the voice result hold in an AGE-RESTRICTED subset (ages 40-80)?
  E. Permutation test: how often does a model trained on SHUFFLED labels score
     as well as the real one?

Evaluation: stratified grouped 5-fold CV (grouped by speaker), repeated 20 times
with different shuffles = 100 test folds. We report the mean and the spread
(standard deviation and 5th-95th percentile) across those folds.

Outputs:  results/step2_cv_results.csv, results/step2_accuracy.png,
          results/step2_feature_importance.csv, models/voice_models.joblib
Run:      .venv/bin/python step2_train_evaluate.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.utils import resample

from features import FEATURE_COLUMNS, RECORDING_COLUMNS
from common import (make_model, grouped_folds, N_REPEATS, N_SPLITS, SEED, style_axes,
                    stamp_prototype, HC_COLOR, PD_COLOR, TEXT, TEXT_2, plt)

OUT, MODELS = Path("results"), Path("models")
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

FEATURE_SETS = {
    "Voice (18 features)": FEATURE_COLUMNS,
    "Recording only (sanity check)": RECORDING_COLUMNS,
    "Age only": ["age"],
    "Sex only": ["sex_male"],
    "Age + sex": ["age", "sex_male"],
}


def evaluate(df: pd.DataFrame, cols, model_name: str) -> dict:
    """Repeated grouped CV. Returns per-fold metric arrays and pooled AUC per repeat."""
    X, y, groups = df[cols].to_numpy(float), df["label"].to_numpy(), df["sample_id"].to_numpy()
    acc, sens, spec, aucs = [], [], [], []
    for rep in range(N_REPEATS):
        oof = np.zeros(len(y))                  # out-of-fold P(Parkinson's) for this repeat
        for train, test in grouped_folds(y, groups, rep):
            model = make_model(model_name).fit(X[train], y[train])
            p = model.predict_proba(X[test])[:, 1]
            oof[test] = p
            pred, truth = (p >= 0.5).astype(int), y[test]
            acc.append(np.mean(pred == truth))
            sens.append(np.mean(pred[truth == 1] == 1))   # Parkinson's correctly flagged
            spec.append(np.mean(pred[truth == 0] == 0))   # controls correctly cleared
        aucs.append(roc_auc_score(y, oof))
    return {"acc": np.array(acc), "sens": np.array(sens), "spec": np.array(spec),
            "auc": np.array(aucs)}


def summarise(res: dict) -> dict:
    """Mean, standard deviation and 5th-95th percentile across the 100 test folds."""
    row = {}
    for k in ("acc", "sens", "spec"):
        v = res[k]
        row[f"{k}_mean"], row[f"{k}_sd"] = v.mean(), v.std()
        row[f"{k}_p5"], row[f"{k}_p95"] = np.percentile(v, 5), np.percentile(v, 95)
    row["auc_mean"], row["auc_sd_across_repeats"] = res["auc"].mean(), res["auc"].std()
    return row


def cv_accuracy(X, y, groups, model_name):
    """Accuracy over one fixed grouped 5-fold split."""
    correct = 0
    for train, test in grouped_folds(y, groups, 0):
        model = make_model(model_name).fit(X[train], y[train])
        correct += np.sum(model.predict(X[test]) == y[test])
    return correct / len(y)


def permutation_p(df, cols, model_name, n_perm):
    """Fraction of label-shuffled models that match or beat the real one (one 5-fold split).

    Labels are shuffled at the SPEAKER level (each speaker keeps a single label).
    Note: sklearn's permutation_test_score(groups=...) shuffles only *within* groups,
    which does nothing when every speaker has one recording, so we do it by hand.
    """
    X, y, groups = df[cols].to_numpy(float), df["label"].to_numpy(), df["sample_id"].to_numpy()
    real = cv_accuracy(X, y, groups, model_name)
    speakers = pd.unique(groups)
    speaker_label = pd.Series(y, index=groups).groupby(level=0).first().loc[speakers].to_numpy()
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(n_perm):
        shuffled = dict(zip(speakers, rng.permutation(speaker_label)))
        y_perm = np.array([shuffled[g] for g in groups])
        null.append(cv_accuracy(X, y_perm, groups, model_name))
    null = np.array(null)
    p = (np.sum(null >= real) + 1) / (n_perm + 1)   # +1: count the real labelling itself
    return real, null.mean(), p


def main():
    df = pd.read_csv(OUT / "features.csv")
    df["sex_male"] = (df["sex"] == "M").astype(int)
    print(f"Chance level (always guess the larger class): {max(df.label.mean(), 1 - df.label.mean()):.3f}")

    # ---- A-C: all participants -------------------------------------------------
    rows = []
    subsets = {"All participants": df,
               "Ages 40-80 only": df[(df.age >= 40) & (df.age <= 80)]}
    for subset_name, d in subsets.items():
        n_hc, n_pd = int((d.label == 0).sum()), int((d.label == 1).sum())
        print(f"\n=== {subset_name}: {n_hc} HC, {n_pd} PD, "
              f"median age HC {d[d.label == 0].age.median():.0f} / PD {d[d.label == 1].age.median():.0f} ===")
        for set_name, cols in FEATURE_SETS.items():
            for model_name in ("logreg", "rf"):
                s = summarise(evaluate(d, cols, model_name))
                rows.append({"subset": subset_name, "n_hc": n_hc, "n_pd": n_pd,
                             "features": set_name, "model": model_name, **s})
                print(f"{set_name:31s} {model_name:6s} acc {s['acc_mean']:.3f} "
                      f"(SD {s['acc_sd']:.3f}, 5-95% {s['acc_p5']:.2f}-{s['acc_p95']:.2f})  "
                      f"sens {s['sens_mean']:.3f}  spec {s['spec_mean']:.3f}  AUC {s['auc_mean']:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "step2_cv_results.csv", index=False)

    # ---- E: permutation tests --------------------------------------------------
    print("\n=== Permutation tests (all participants, one 5-fold split) ===")
    for set_name in ("Voice (18 features)", "Recording only (sanity check)"):
        for model_name, n_perm in (("logreg", 1000), ("rf", 100)):
            score, null_mean, p = permutation_p(df, FEATURE_SETS[set_name], model_name, n_perm)
            print(f"{set_name:31s} {model_name:6s} accuracy {score:.3f}  "
                  f"shuffled-label mean {null_mean:.3f}  p = {p:.3f}  ({n_perm} shuffles)")

    # ---- Which features does the voice model lean on? -----------------------------
    # Permutation importance on held-out folds: how much does accuracy drop when one
    # feature is shuffled? Averaged over one 5-fold split.
    X, y, g = df[FEATURE_COLUMNS].to_numpy(float), df.label.to_numpy(), df.sample_id.to_numpy()
    imp = {m: np.zeros(len(FEATURE_COLUMNS)) for m in ("logreg", "rf")}
    for m in imp:
        for train, test in grouped_folds(y, g, 0):
            model = make_model(m).fit(X[train], y[train])
            r = permutation_importance(model, X[test], y[test], n_repeats=30,
                                       random_state=SEED, scoring="accuracy")
            imp[m] += r.importances_mean / N_SPLITS
    imp_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "logreg": imp["logreg"], "rf": imp["rf"],
                           "spearman_with_age": [df[c].corr(df.age, method="spearman")
                                                 for c in FEATURE_COLUMNS]})
    imp_df = imp_df.sort_values("logreg", ascending=False)
    imp_df.to_csv(OUT / "step2_feature_importance.csv", index=False)
    print("\nTop held-out permutation importances (accuracy drop):")
    print(imp_df.head(6).round(3).to_string(index=False))

    # ---- Chart: accuracy by feature set, with fold spread -------------------------
    plot_accuracy(res)

    # ---- Final models for the app (trained on ALL 81 recordings) -------------------
    # A bootstrap ensemble gives the app a score RANGE instead of one falsely
    # precise number: each member is trained on a resampled copy of the data, and
    # the spread of their outputs shows how much the score depends on which
    # training speakers happened to be in the sample.
    ensemble = {}
    for m in ("logreg", "rf"):
        members = []
        for b in range(100 if m == "logreg" else 30):
            Xb, yb = resample(X, y, stratify=y, random_state=SEED + b)
            members.append(make_model(m).fit(Xb, yb))
        ensemble[m] = {"full": make_model(m).fit(X, y), "bootstrap": members}
    joblib.dump({"feature_columns": FEATURE_COLUMNS, "models": ensemble,
                 "cv_results": res}, MODELS / "voice_models.joblib")
    print(f"\nSaved final models to {MODELS / 'voice_models.joblib'}")


def plot_accuracy(res: pd.DataFrame):
    """Horizontal bars: mean CV accuracy per feature set, whiskers = 5th-95th pct of folds."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, subset in zip(axes, ["All participants", "Ages 40-80 only"]):
        r = res[res.subset == subset]
        names = list(FEATURE_SETS)
        ypos = np.arange(len(names))
        for k, (model, color, label) in enumerate([("logreg", HC_COLOR, "Logistic regression"),
                                                   ("rf", PD_COLOR, "Random forest")]):
            rr = r[r.model == model].set_index("features").loc[names]
            yy = ypos + (k - 0.5) * 0.36
            ax.barh(yy, rr.acc_mean, height=0.32, color=color, label=label)
            ax.errorbar(rr.acc_mean, yy, xerr=[rr.acc_mean - rr.acc_p5, rr.acc_p95 - rr.acc_mean],
                        fmt="none", ecolor=TEXT_2, elinewidth=1, capsize=2)
            for yv, v in zip(yy, rr.acc_mean):
                ax.text(0.02, yv, f"{v:.2f}", va="center", fontsize=8, color="white")
        ax.axvline(0.5, color=TEXT, linestyle="--", linewidth=1)
        ax.text(0.51, len(names) - 0.55, "chance", fontsize=8, color=TEXT, va="center")
        ax.set_yticks(ypos, names)
        ax.set_xlim(0, 1)
        n = r.iloc[0]
        ax.set_title(f"{subset}  ({n.n_hc} HC, {n.n_pd} PD)", fontsize=10, color=TEXT, loc="left")
        style_axes(ax)
        ax.grid(axis="x", color="#e6e6e6"); ax.grid(axis="y", visible=False)
    axes[0].set_ylim(len(FEATURE_SETS) - 0.3, -0.7)   # shared y-axis: flip once, first set on top
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2, fontsize=8, frameon=False)
    fig.supxlabel("Cross-validated accuracy (bars = mean, whiskers = 5th-95th percentile of 100 test folds)",
                  fontsize=9, color=TEXT_2)
    fig.suptitle("Step 2: who can tell Parkinson's from healthy, and from what?",
                 x=0.01, ha="left", fontsize=12, color=TEXT)
    stamp_prototype(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "step2_accuracy.png", dpi=150)


# Re-draw the chart from saved results without re-running the analysis:
#   .venv/bin/python -c "import step2_train_evaluate as s, pandas as pd; s.plot_accuracy(pd.read_csv('results/step2_cv_results.csv'))"


if __name__ == "__main__":
    main()
