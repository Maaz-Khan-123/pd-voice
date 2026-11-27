"""
Step 4: covariate analysis. Does the voice add anything beyond age and sex?

RESEARCH PROTOTYPE, NOT A DIAGNOSIS.

Uses the same speaker-grouped, stratified 5-fold cross-validation as step 2
(20 repeats = 100 test folds) and the same model settings (common.py). Existing
features, models and results are only READ, never changed.

  Model A: age + sex only
  Model B: voice features only (the 18 features used by the app)
  Model C: age + sex + voice features

Each is run on:
  * all speakers,
  * ages 55-80: the age window where the two groups' ages are NOT significantly
    different (chosen by searching windows in 5-year steps; see the printout),
  * ages 40-80: the window used in step 2, for comparison (ages still differ there).

Because the age-matched subset is unbalanced (few healthy speakers), we report
plain accuracy AND balanced accuracy (mean of sensitivity and specificity), plus
the accuracy of always guessing the larger group.

Models A and C are compared on IDENTICAL folds, so "C minus A" is a paired
difference: it answers "does adding voice to age help?" directly.

Outputs: results/step4_covariates.csv (one row per subset x model x classifier)
         results/step4_covariates.json (same numbers + paired comparisons + subset info)
Run:     .venv/bin/python step4_covariates.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from features import FEATURE_COLUMNS
from common import make_model, grouped_folds, N_REPEATS

OUT = Path("results")

FEATURE_SETS = {
    "A: age + sex": ["age", "sex_male"],
    "B: voice only": FEATURE_COLUMNS,
    "C: age + sex + voice": ["age", "sex_male"] + FEATURE_COLUMNS,
}
SUBSETS = {
    "All speakers": (0, 200),
    "Age-matched 55-80": (55, 80),
    "Ages 40-80 (not matched)": (40, 80),
}


def fold_scores(d: pd.DataFrame, cols, model_name):
    """Accuracy and balanced accuracy on every test fold (100 folds, fixed seeds)."""
    X, y, g = d[cols].to_numpy(float), d["label"].to_numpy(), d["sample_id"].to_numpy()
    acc, bacc = [], []
    for rep in range(N_REPEATS):
        for tr, te in grouped_folds(y, g, rep):
            pred = make_model(model_name).fit(X[tr], y[tr]).predict(X[te])
            truth = y[te]
            acc.append(np.mean(pred == truth))
            sens = np.mean(pred[truth == 1] == 1)
            spec = np.mean(pred[truth == 0] == 0)
            bacc.append((sens + spec) / 2)
    return np.array(acc), np.array(bacc)


def summarise(v):
    return {"mean": float(v.mean()), "sd": float(v.std()),
            "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}


def main():
    df = pd.read_csv(OUT / "features.csv")
    df["sex_male"] = (df["sex"] == "M").astype(int)

    rows, paired, subset_info = [], [], {}
    for subset, (lo, hi) in SUBSETS.items():
        d = df[(df.age >= lo) & (df.age <= hi)].reset_index(drop=True)
        hc, pd_ = d[d.label == 0], d[d.label == 1]
        info = {"age_range": [lo, hi] if hi < 200 else None,
                "n_healthy": int(len(hc)), "n_parkinsons": int(len(pd_)),
                "median_age_healthy": float(hc.age.median()),
                "median_age_parkinsons": float(pd_.age.median()),
                "age_difference_p": float(mannwhitneyu(hc.age, pd_.age).pvalue),
                "chance_accuracy": float(max(len(hc), len(pd_)) / len(d))}
        subset_info[subset] = info
        print(f"\n=== {subset}: {info['n_healthy']} healthy, {info['n_parkinsons']} Parkinson's | "
              f"median age {info['median_age_healthy']:.0f} vs {info['median_age_parkinsons']:.0f} "
              f"(p = {info['age_difference_p']:.3g}) | always-guess-larger-group = "
              f"{info['chance_accuracy']:.1%} ===")

        for model_name in ("logreg", "rf"):
            per_fold = {}
            for set_name, cols in FEATURE_SETS.items():
                acc, bacc = fold_scores(d, cols, model_name)
                per_fold[set_name] = (acc, bacc)
                rows.append({"subset": subset, "classifier": model_name, "model": set_name,
                             "n_healthy": info["n_healthy"], "n_parkinsons": info["n_parkinsons"],
                             "chance_accuracy": info["chance_accuracy"],
                             **{f"acc_{k}": v for k, v in summarise(acc).items()},
                             **{f"bal_acc_{k}": v for k, v in summarise(bacc).items()}})
                print(f"  {model_name:6s} {set_name:22s} accuracy {acc.mean():.3f} (SD {acc.std():.3f}, "
                      f"5-95% {np.percentile(acc, 5):.2f}-{np.percentile(acc, 95):.2f})   "
                      f"balanced {bacc.mean():.3f} (SD {bacc.std():.3f})")

            # Paired: same folds, so the per-fold difference isolates the effect of adding voice.
            for metric, idx in (("accuracy", 0), ("balanced_accuracy", 1)):
                diff = per_fold["C: age + sex + voice"][idx] - per_fold["A: age + sex"][idx]
                rep_means = diff.reshape(N_REPEATS, -1).mean(axis=1)   # one value per repeat
                paired.append({"subset": subset, "classifier": model_name, "metric": metric,
                               "C_minus_A_mean": float(diff.mean()),
                               "C_minus_A_fold_sd": float(diff.std()),
                               "repeats_where_C_beats_A": int((rep_means > 0).sum()),
                               "repeats_where_C_worse_than_A": int((rep_means < 0).sum()),
                               "n_repeats": N_REPEATS})
            p_acc = [x for x in paired if x["subset"] == subset and x["classifier"] == model_name][0]
            print(f"  {model_name:6s} adding voice to age+sex (C - A): accuracy {p_acc['C_minus_A_mean']:+.3f} "
                  f"(fold SD {p_acc['C_minus_A_fold_sd']:.3f}); C better in "
                  f"{p_acc['repeats_where_C_beats_A']}/{N_REPEATS} repeats, worse in "
                  f"{p_acc['repeats_where_C_worse_than_A']}/{N_REPEATS}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "step4_covariates.csv", index=False)

    # ---- Plain verdict (logistic regression, the app's default model) ----------
    verdicts = {}
    for subset in SUBSETS:
        a = res[(res.subset == subset) & (res.classifier == "logreg") & (res.model == "A: age + sex")].iloc[0]
        c = res[(res.subset == subset) & (res.classifier == "logreg") & (res.model == "C: age + sex + voice")].iloc[0]
        pa = [x for x in paired if x["subset"] == subset and x["classifier"] == "logreg"
              and x["metric"] == "balanced_accuracy"][0]
        diff = pa["C_minus_A_mean"]
        helps = diff > 0.02 and pa["repeats_where_C_beats_A"] >= 0.8 * N_REPEATS
        hurts = diff < -0.02 and pa["repeats_where_C_worse_than_A"] >= 0.8 * N_REPEATS
        # Beating Model A means nothing if Model C itself is no better than chance
        # (balanced accuracy 0.5). Within a narrow age range, age carries no signal and
        # Model A can fall BELOW chance, so "C > A" there is not evidence for the voice.
        c_above_chance = c["bal_acc_mean"] > 0.55
        if helps and not c_above_chance:
            verdicts[subset] = (f"no evidence: C beats A ({a['bal_acc_mean']:.2f}) only because A is "
                                f"below chance; C itself ({c['bal_acc_mean']:.2f}) is at chance level")
        else:
            verdicts[subset] = ("improves" if helps else "makes it worse" if hurts else "no clear change")
    json.dump({"description": "Step 4 covariate analysis (research prototype, not a diagnosis)",
               "cv": f"stratified speaker-grouped 5-fold x {N_REPEATS} repeats",
               "subsets": subset_info, "results": rows, "paired_C_vs_A": paired,
               "verdict_adding_voice_to_age_logreg_balanced_accuracy": verdicts},
              open(OUT / "step4_covariates.json", "w"), indent=2)

    print("\n=== Does adding voice to age (+ sex) help? (logistic regression, balanced accuracy) ===")
    for subset, v in verdicts.items():
        pa = [x for x in paired if x["subset"] == subset and x["classifier"] == "logreg"
              and x["metric"] == "balanced_accuracy"][0]
        print(f"  {subset:26s}: {v.upper():16s} (C - A = {pa['C_minus_A_mean']:+.3f}; C better in "
              f"{pa['repeats_where_C_beats_A']}/{N_REPEATS} repeats)")
    print(f"\nSaved {OUT / 'step4_covariates.csv'} and {OUT / 'step4_covariates.json'}")


if __name__ == "__main__":
    main()
