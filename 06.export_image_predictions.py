import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from cifar10_data import PROJECT, load_data
from gpu_check import check_gpu
from image_features import extract_features, load_encoder

SOURCE = PROJECT / "results" / "comparison_20261119_225145_293414"
MODELS = ["resnet18", "mobilenet_v3_small"]
POLICIES = ["Random", "Entropy"]
SEEDS = [11]
BUDGET = 20
SCORE_TOLERANCE = 1e-8


def training_positions(model_name, seed, policy, pool, expected_count):
    path = SOURCE / f"{model_name}_{seed}_{policy}_{BUDGET}_ids.npy"
    ids = np.load(path, allow_pickle=False)
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) != expected_count
            or len(np.unique(ids)) != expected_count or not np.isin(ids, pool).all()):
        raise ValueError(f"Invalid training IDs in {path}")
    # Preserve the archived row order when fitting the classifier.
    return np.searchsorted(pool, ids)


def prediction_table(head, clean, noisy, image_ids, labels, names):
    table = pd.DataFrame({"test_id": image_ids, "true_label": labels,
                          "true_class": names[labels]})
    for domain, features in [("clean", clean), ("noisy", noisy)]:
        probabilities = head.predict_proba(features)
        predicted = head.classes_[probabilities.argmax(axis=1)]
        table[f"{domain}_pred"] = predicted
        table[f"{domain}_class"] = names[predicted]
        table[f"{domain}_confidence"] = probabilities.max(axis=1)
        table[f"{domain}_correct"] = predicted == labels
        for column, class_id in enumerate(head.classes_):
            table[f"{domain}_p_{names[class_id]}"] = probabilities[:, column]
    return table


def compare_scores(table, reference):
    rows = []
    for column, domain in [("clean_pred", "clean"), ("noisy_pred", "noise")]:
        predicted = table[column]
        scores = {"acc": accuracy_score(table.true_label, predicted),
                  "f1": f1_score(table.true_label, predicted, average="macro")}
        for metric, value in scores.items():
            original = reference.loc[domain, metric]
            difference = value - original
            rows.append(dict(domain=domain, metric=metric, original=original,
                             exported=value, difference=difference,
                             matches=abs(difference) <= SCORE_TOLERANCE))
    return pd.DataFrame(rows)


def run():
    check_gpu()
    metrics = pd.read_csv(SOURCE / "metrics.csv")
    settings = json.loads((SOURCE / "environment.json").read_text(encoding="utf-8"))
    train, test = load_data()
    train_labels = np.array(train.targets)
    test_labels = np.array(test.targets)
    pool = np.flatnonzero(train_labels < 8)
    known = np.flatnonzero(test_labels < 8)
    names = np.array(test.classes)
    batch = settings["batch_size"]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = PROJECT / "results" / f"image_predictions_{stamp}"
    folder.mkdir(parents=True)
    print("Output:", folder, flush=True)
    output = folder / "image_predictions.csv"
    checks = []

    for model_name in MODELS:
        print("Extracting features:", model_name, flush=True)
        model, preprocess = load_encoder(model_name, "cuda")
        train_x = extract_features(model, preprocess, train, pool, "cuda", batch)
        clean = extract_features(model, preprocess, test, known, "cuda", batch)
        noisy = extract_features(model, preprocess, test, known, "cuda", batch, noisy=True)
        for seed in SEEDS:
            for policy in POLICIES:
                print(f"{model_name} | {policy} | seed {seed} | {BUDGET}%", flush=True)
                ref = metrics.query("arch == @model_name and seed == @seed and policy == @policy and budget == @BUDGET")
                if sorted(ref.domain) != ["clean", "noise"]:
                    raise ValueError("Expected one clean and one noise row for this configuration.")
                ref = ref.set_index("domain")
                positions = training_positions(model_name, seed, policy, pool, int(ref.loc["clean", "n"]))
                head = LogisticRegression(C=1, solver=ref.loc["clean", "solver"],
                                          max_iter=settings["max_iter"])
                head.fit(train_x[positions], train_labels[pool[positions]])
                table = prediction_table(head, clean, noisy, known, test_labels[known], names)
                config = dict(arch=model_name, seed=seed, policy=policy, budget=BUDGET)
                table.assign(**config).to_csv(output, mode="a", header=not output.exists(), index=False)
                check = compare_scores(table, ref).assign(**config)
                checks.append(check)
                pd.concat(checks, ignore_index=True).to_csv(folder / "score_check.csv", index=False)
                print("Score check:", "PASS" if check.matches.all() else "MISMATCH", flush=True)
    print("Saved:", output)


if __name__ == "__main__":
    run()
