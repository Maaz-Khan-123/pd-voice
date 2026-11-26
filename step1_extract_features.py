"""
Step 1: run extract_features() on every training recording and save a CSV.

RESEARCH PROTOTYPE, NOT A DIAGNOSIS.

Output: results/features.csv, one row per recording with
  sample_id, label (0 = healthy control, 1 = Parkinson's), age, sex,
  the voice features, quality-control columns, and recording-only
  "sanity check" features (rec_*), which are never used by the voice model.
Run:  .venv/bin/python step1_extract_features.py
"""

from pathlib import Path
import pandas as pd
from features import (extract_features, recording_features,
                      FEATURE_COLUMNS, QC_COLUMNS, RECORDING_COLUMNS)

DATA = Path("data/figshare")
OUT = Path("results")
OUT.mkdir(exist_ok=True)


def load_metadata() -> pd.DataFrame:
    """Demographics sheet + the path of each WAV. Sample ID == file name without .wav."""
    demo = pd.read_excel(DATA / "Demographics_age_sex.xlsx", sheet_name="Parselmouth")
    demo = demo.rename(columns={"Sample ID": "sample_id", "Age": "age", "Sex": "sex"})
    demo["label"] = (demo["Label"] == "PwPD").astype(int)
    folder = {0: DATA / "HC_AH" / "HC_AH", 1: DATA / "PD_AH" / "PD_AH"}
    demo["path"] = [folder[l] / f"{s}.wav" for s, l in zip(demo.sample_id, demo.label)]
    missing = [p for p in demo.path if not p.exists()]
    assert not missing, f"Missing WAV files: {missing}"
    return demo[["sample_id", "label", "age", "sex", "path"]]


def main():
    meta = load_metadata()
    rows = []
    for i, r in meta.iterrows():
        feats = extract_features(r.path)          # voice features (model inputs)
        rec = recording_features(r.path)          # recording-only features (sanity check)
        rows.append({"sample_id": r.sample_id, "label": r.label, "age": r.age,
                     "sex": r.sex, **feats, **rec})
        print(f"[{i + 1:2d}/{len(meta)}] {r.sample_id[:16]}  "
              f"F0={feats['f0_mean_hz']:6.1f} Hz  HNR={feats['hnr_db']:5.1f} dB")
    df = pd.DataFrame(rows)[["sample_id", "label", "age", "sex"] + FEATURE_COLUMNS + QC_COLUMNS
                              + RECORDING_COLUMNS]
    df.to_csv(OUT / "features.csv", index=False)
    print(f"\nSaved {len(df)} rows x {len(FEATURE_COLUMNS)} features to {OUT / 'features.csv'}")
    print("Rows with any missing feature:", int(df[FEATURE_COLUMNS].isna().any(axis=1).sum()))


if __name__ == "__main__":
    main()
