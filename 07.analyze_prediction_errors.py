from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

PROJECT = Path(__file__).resolve().parent
RUN = "image_predictions_20261120_021017_526692"
SOURCE = PROJECT / "results" / RUN / "image_predictions.csv"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse"]
GROUP = ["arch", "policy", "seed", "budget"]


def class_scores(group):
    truth = group["true_class"]
    clean = group["clean_class"]
    noisy = group["noisy_class"]
    rows = []
    clean_f1 = f1_score(truth, clean, labels=CLASSES, average=None, zero_division=0)
    noisy_f1 = f1_score(truth, noisy, labels=CLASSES, average=None, zero_division=0)
    for index, name in enumerate(CLASSES):
        actual = truth == name
        correct_before = clean == name
        correct_after = noisy == name
        drop = 100 * (clean_f1[index] - noisy_f1[index])
        rows.append({
            "class": name, "support": actual.sum(),
            "clean_f1": clean_f1[index], "noisy_f1": noisy_f1[index],
            "f1_drop_pp": drop, "macro_drop_contribution_pp": drop / len(CLASSES),
            "correct_to_wrong": (actual & correct_before & ~correct_after).sum(),
            "wrong_to_correct": (actual & ~correct_before & correct_after).sum(),
            "predicted_clean": correct_before.sum(), "predicted_noisy": correct_after.sum(),
        })
    return pd.DataFrame(rows).sort_values("f1_drop_pp", ascending=False)


def save_matrices(group, folder, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for domain, ax in zip(["clean", "noisy"], axes):
        counts = confusion_matrix(group.true_class, group[f"{domain}_class"], labels=CLASSES)
        table = pd.DataFrame(counts, index=CLASSES, columns=CLASSES)
        table.to_csv(folder / f"confusion_{domain}.csv", index_label="true_class")
        percent = 100 * counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(percent, vmin=0, vmax=100, cmap="Blues")
        for row, col in np.ndindex(counts.shape):
            color = "white" if percent[row, col] > 55 else "black"
            ax.text(col, row, f"{percent[row, col]:.1f}", ha="center", va="center", color=color)
        ax.set_xticks(range(8), CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(8), CLASSES)
        ax.set(title=domain.capitalize(), xlabel="Predicted class", ylabel="True class")
    fig.suptitle(f"{title}\nPercentage within each true class")
    fig.savefig(folder / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def run():
    data = pd.read_csv(SOURCE)
    if data.duplicated(GROUP + ["test_id"]).any():
        raise ValueError("Duplicate image IDs within a configuration.")
    for column in ["true_class", "clean_class", "noisy_class"]:
        if not data[column].isin(CLASSES).all():
            raise ValueError(f"Unexpected class in {column}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = PROJECT / "results" / f"prediction_errors_{stamp}"
    output.mkdir(parents=True)
    summary = []
    for (model, policy, seed, budget), group in data.groupby(GROUP):
        if set(group.true_class) != set(CLASSES):
            raise ValueError("One or more test classes are missing.")
        title = f"{model} | {policy} | seed {seed} | {budget}%"
        folder = output / f"{model}_{policy}_{seed}_{budget}"
        folder.mkdir()
        scores = class_scores(group)
        scores.to_csv(folder / "class_scores.csv", index=False)
        save_matrices(group, folder, title)

        lost = group[(group.clean_class == group.true_class) & (group.noisy_class != group.true_class)]
        columns = ["test_id", "true_class", "clean_class", "clean_confidence", "noisy_class", "noisy_confidence"]
        examples = lost.sort_values(["noisy_confidence", "test_id"], ascending=[False, True])
        examples.groupby("true_class").head(3)[columns].to_csv(folder / "error_examples.csv", index=False)

        summary.append(dict(arch=model, policy=policy, seed=seed, budget=budget,
                            images=len(group), clean_macro_f1=scores.clean_f1.mean(),
                            noisy_macro_f1=scores.noisy_f1.mean(),
                            macro_f1_drop_pp=scores.macro_drop_contribution_pp.sum(),
                            correct_to_wrong=len(lost), wrong_to_correct=scores.wrong_to_correct.sum(),
                            largest_f1_drop_class=scores.iloc[0]["class"]))
        print(title, "| largest F1 drop:", scores.iloc[0]["class"])
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    print("Saved in:", output)


if __name__ == "__main__":
    run()
