"""
common.py: shared model definitions, cross-validation and chart styling.

RESEARCH PROTOTYPE, NOT A DIAGNOSIS.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")                        # draw to files, no window needed
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

SEED = 0
N_SPLITS = 5          # 5 folds: each test fold holds ~16 different speakers
N_REPEATS = 20        # repeat with 20 different shufflings -> 100 test folds

# Chart colours (validated reference palette, slots 1 and 2) and text inks.
HC_COLOR, PD_COLOR = "#257de0", "#f47230"
TEXT, TEXT_2, GRID = "#191919", "#666666", "#e6e6e6"
PROTOTYPE_LABEL = "Research prototype, not a diagnosis"


def make_model(name: str):
    """The two baseline classifiers. Settings are fixed in advance, not tuned on this data."""
    if name == "logreg":
        # Features have very different units (Hz, dB, ratios), so standardise first.
        # The scaler is inside the pipeline, so it is re-fitted on each training fold
        # and never sees the test speakers.
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                      class_weight="balanced", random_state=SEED, n_jobs=-1)
    raise ValueError(name)


def grouped_folds(y, groups, repeat: int):
    """Stratified, GROUPED 5-fold split: all recordings of one speaker stay in the same fold.

    In this dataset each speaker has one recording, but grouping by sample_id keeps
    the code correct if repeat recordings are ever added.
    """
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat)
    return cv.split(np.zeros(len(y)), y, groups)


def style_axes(ax):
    """Recessive grid and axes so the data stand out."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.yaxis.label.set_color(TEXT_2)
    ax.xaxis.label.set_color(TEXT_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def stamp_prototype(fig):
    """Every saved figure carries the prototype label."""
    fig.text(0.99, 0.01, PROTOTYPE_LABEL, ha="right", va="bottom", fontsize=8,
             color=TEXT_2, style="italic")
