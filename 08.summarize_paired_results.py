from figure_assets import copy_figures
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
RUN = "comparison_20261119_225145_293414"
SOURCE = PROJECT / "results" / RUN / "metrics.csv"
MODELS = ["resnet18", "mobilenet_v3_small"]
BUDGETS = [10, 20]
DOMAINS = ["clean", "noise"]
SEEDS = [11, 22, 33, 44, 55]


def pair_scores(data):
    keys = ["arch", "budget", "domain", "seed"]
    expected = pd.MultiIndex.from_product([MODELS, BUDGETS, DOMAINS, SEEDS], names=keys)
    rows = data[data.policy.isin(["Random", "Entropy"])].set_index(keys)
    rows = rows[rows.index.isin(expected)].reset_index()
    if rows.duplicated(keys + ["policy"]).any():
        raise ValueError("Duplicate scores for the same seed, model, domain, budget and policy.")
    scores = rows.pivot(index=keys, columns="policy", values="f1")
    scores = scores.reindex(index=expected, columns=["Random", "Entropy"])
    missing = scores[scores.isna().any(axis=1)]
    if not missing.empty:
        raise ValueError(f"Missing paired F1 scores:\n{missing}")
    if not ((scores >= 0) & (scores <= 1)).all().all():
        raise ValueError("Expected F1 scores between 0 and 1.")
    scores["delta_pp"] = 100 * (scores["Entropy"] - scores["Random"])
    return scores.reset_index()


def summarize(pairs):
    rows = []
    for (model, budget, domain), group in pairs.groupby(["arch", "budget", "domain"]):
        delta = group.delta_pp
        rows.append(dict(arch=model, budget=budget, domain=domain, seeds=len(delta),
                         mean_delta_pp=delta.mean(), sd_delta_pp=delta.std(ddof=1),
                         min_delta_pp=delta.min(), max_delta_pp=delta.max(),
                         entropy_wins=(delta > 0).sum(), ties=(delta == 0).sum(),
                         random_wins=(delta < 0).sum()))
    return pd.DataFrame(rows)


def plot_differences(pairs, path):
    fig, axes = plt.subplots(len(MODELS), len(DOMAINS), squeeze=False, sharey=True,
                             figsize=(10, 6.5), layout="constrained")
    for row, model in enumerate(MODELS):
        for col, domain in enumerate(DOMAINS):
            ax = axes[row, col]
            for position, budget in enumerate(BUDGETS):
                group = pairs.query("arch == @model and domain == @domain and budget == @budget")
                offsets = np.linspace(-0.13, 0.13, len(group))
                ax.scatter(position + offsets, group.delta_pp, color="#126a84", s=38)
                ax.hlines(group.delta_pp.mean(), position - 0.2, position + 0.2, color="#be4b19")
            ax.axhline(0, lw=1, ls="--", color="gray")
            ax.set_xticks(range(len(BUDGETS)), [f"{b}%" for b in BUDGETS])
            ax.set_xlim(-0.5, len(BUDGETS) - 0.5)
            ax.set_title(f"{model} | {domain}")
            if col == 0:
                ax.set_ylabel("Entropy minus Random (pp)")
    fig.suptitle("Paired macro-F1 by label budget\nEach dot is one seed; orange bars show means")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run():
    pairs = pair_scores(pd.read_csv(SOURCE))
    summary = summarize(pairs)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = PROJECT / "results" / f"paired_results_{stamp}"
    folder.mkdir(parents=True)
    pairs.to_csv(folder / "paired_by_seed.csv", index=False)
    summary.to_csv(folder / "paired_summary.csv", index=False)
    plot_differences(pairs, folder / "paired_differences.png")
    print(summary.round(3).to_string(index=False))
    copy_figures(__file__, folder)
    print("Saved in:", folder)


if __name__ == "__main__":
    run()
