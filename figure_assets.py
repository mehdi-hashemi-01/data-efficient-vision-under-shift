from datetime import datetime
from pathlib import Path
from shutil import copy2
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TEAL = "#087E83"
ORANGE = "#CF603B"
MODELS = {"resnet18": "ResNet18", "mobilenet_v3_small": "MobileNetV3-Small"}


def asset_dir(script, run):
    folder = ROOT / "assets" / Path(script).stem / run
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def copy_examples(script, image):
    stamp = datetime.now().strftime("examples_%Y%m%d_%H%M%S_%f")
    copy2(image, asset_dir(script, stamp) / image.name)


def copy_figures(script, run):
    dest = asset_dir(script, run.name)
    for image in run.rglob("*.png"):
        target = dest / image.relative_to(run)
        target.parent.mkdir(parents=True, exist_ok=True)
        copy2(image, target)


def finish(fig, script, run, filename):
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#BAC7C8")
        ax.tick_params(colors="#344D55", length=3)
    fig.tight_layout(pad=1.4)
    fig.savefig(asset_dir(script, run.name) / filename, dpi=165, facecolor="white")
    plt.close(fig)


def baseline_chart(script, run):
    data = pd.read_csv(run / "baseline_scores.csv").set_index("domain")
    ax = data[["accuracy", "macro_f1"]].rename(columns={"accuracy": "Accuracy", "macro_f1": "Macro-F1"}).plot.bar(
        color=[TEAL, ORANGE], rot=0, ylim=(0, 1), figsize=(7, 3.8))
    ax.set(xlabel="", ylabel="Score", title="First baseline on paired test images")
    ax.legend(frameon=False)
    finish(ax.figure, script, run, "baseline.png")


def budget_chart(script, run):
    data = pd.read_csv(run / "metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, domain in zip(axes, ["clean", "noise"]):
        table = data[data.domain == domain]
        for (arch, policy), group in table.groupby(["arch", "policy"]):
            scores = group.groupby("budget").f1.mean()
            ax.plot(scores.index, scores, marker="o", ls="-" if policy == "Entropy" else "--",
                    color=TEAL if arch == "resnet18" else ORANGE, label=f"{MODELS[arch]} · {policy}")
        ax.set(title="Clean test" if domain == "clean" else "Gaussian noise", xlabel="Label budget (%)",
               ylim=(0, 1), xticks=[5, 10, 20, 100])
    axes[0].set_ylabel("Mean macro-F1 across seeds")
    axes[1].legend(frameon=False, fontsize=8)
    finish(fig, script, run, "budget_curves.png")


def transitions_chart(script, run):
    data = pd.read_csv(run / "image_predictions.csv")
    data["Correct → wrong"] = data.clean_correct & ~data.noisy_correct
    data["Wrong → correct"] = ~data.clean_correct & data.noisy_correct
    groups = ["arch", "policy", "seed", "budget"]
    counts = data.groupby(groups)[["Correct → wrong", "Wrong → correct"]].sum()
    counts.index = [f"{MODELS[m]} · {p} · seed {s} · {b}%" for m,p,s,b in counts.index]
    ax = counts.plot.barh(color=[ORANGE, TEAL], figsize=(9, max(3.7, len(counts)*0.55)))
    ax.set(title="How predictions change after adding noise", xlabel="Test images per configuration", ylabel="")
    ax.legend(frameon=False)
    finish(ax.figure, script, run, "transitions.png")


def split_chart(script, run):
    counts = json.loads((run / "split_summary.json").read_text())["validation_class_counts"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.bar(counts.keys(), counts.values(), color=TEAL)
    ax.bar_label(bars, padding=3)
    ax.margins(y=0.15)
    ax.set(title="Class counts in the fixed validation split", ylabel="Images")
    finish(fig, script, run, "validation_classes.png")


def validation_chart(script, run):
    data = pd.read_csv(run / "metrics.csv")
    table = data[data.budget == 20].pivot_table(index=["arch", "policy"], columns="domain", values="f1")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, ((model, policy), row) in enumerate(table.iterrows()):
        ax.plot([row.noise, row.clean], [i, i], color="#CAD6D5", lw=3, zorder=1)
        for domain, color, offset in [("noise", ORANGE, -0.028), ("clean", TEAL, 0.028)]:
            ax.scatter(row[domain], i, s=68, color=color, label=domain.title() if i == 0 else None)
            ax.text(row[domain]+offset, i, f"{row[domain]:.3f}", va="center",
                    ha="right" if offset < 0 else "left", fontsize=9)
    ax.set_yticks(np.arange(len(table)), [f"{MODELS[m]} · {p}" for m,p in table.index])
    ax.set(xlim=(0,1), ylim=(-0.7,len(table)-0.05), xlabel="Mean validation macro-F1",
           title="Clean and noisy views of the same 400 images")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    finish(fig, script, run, "validation_macro_f1.png")
