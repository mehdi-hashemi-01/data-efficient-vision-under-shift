import hashlib
import json
import platform
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from evaluation_metrics import assess

ROOT = Path(__file__).resolve().parent
SPLIT = ROOT / "results" / "validation_split_v1"
MODELS = ["resnet18", "mobilenet_v3_small"]
SEEDS = [11, 22, 33, 44, 55]
TRAIN_COUNTS = {5: 1600, 10: 3600, 20: 7600, 100: 39600}
BATCH_SIZE = 32


def read_split(labels):
    summary = json.loads((SPLIT / "split_summary.json").read_text(encoding="utf-8"))
    known = np.flatnonzero(np.isin(labels, np.arange(8)))
    draw = np.random.Generator(np.random.PCG64(20261120)).permutation(known)
    arrays = []
    for name, expected in [("validation_ids.npy", draw[:400]), ("pool_ids.npy", draw[400:])]:
        path = SPLIT / name
        ids = np.load(path, allow_pickle=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != summary["sha256"][name] or not np.array_equal(ids, np.sort(expected)):
            raise ValueError(f"Split verification failed for {name}. Check the output of script 09.")
        arrays.append(ids)
    if len(known) != 40000 or len(np.unique(labels[arrays[0]])) != 8:
        raise ValueError("Unexpected size or class coverage for the CIFAR-10 split.")
    return arrays[0], arrays[1], summary


def fit_head(x, y):
    retries = []
    for solver in ("lbfgs", "newton-cg"):
        head = LogisticRegression(C=1, solver=solver, max_iter=3000)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                head.fit(x, y)
            return head, " | ".join(retries)
        except ConvergenceWarning as warning:
            retries.append(f"{solver}: {warning}")
            print(retries[-1], flush=True)
    raise RuntimeError("Both solvers reached their convergence limit.")


def add_uncertain(x, selected, count, head):
    remaining = np.setdiff1d(np.arange(len(x)), selected)
    prob = head.predict_proba(x[remaining])
    entropy = -(prob * np.log(prob.clip(min=1e-15))).sum(axis=1)
    # Pool rows follow ascending original image ID, including ties.
    ranked = np.argsort(-entropy, kind="stable")
    return np.concatenate([selected, remaining[ranked[:count - len(selected)]]])


def run_encoder(name, pool, x, labels, validation, views, folder, rows):
    def evaluate(selected, budget, seed, policy, acquisition_seconds):
        if len(selected) != TRAIN_COUNTS[budget] or len(np.unique(selected)) != len(selected):
            raise ValueError("Unexpected training subset.")
        ids = pool[selected]
        start = perf_counter()
        head, warning = fit_head(x[selected], labels[ids])
        fit_seconds = perf_counter() - start
        if not np.array_equal(head.classes_, np.arange(8)):
            raise ValueError("Training subset does not cover all eight classes.")
        tag = f"{name}_{seed}_{policy}_{budget}"
        np.save(folder / f"{tag}_ids.npy", ids)
        predictions = {"train_image_id": validation, "true_class": labels[validation]}
        for domain, features in views.items():
            prob = head.predict_proba(features)
            predictions[f"{domain}_prediction"] = prob.argmax(axis=1)
            predictions[f"{domain}_confidence"] = prob.max(axis=1)
            rows.append(dict(arch=name, seed=seed, policy=policy, budget=budget,
                             n_train=len(ids), n_validation=len(validation), domain=domain,
                             solver=head.solver, iterations=int(head.n_iter_.max()),
                             convergence_warning=warning, acquisition_seconds=acquisition_seconds,
                             fit_seconds=fit_seconds, **assess(prob, labels[validation])))
        pd.DataFrame(predictions).to_csv(folder / f"{tag}_predictions.csv", index=False)
        temp = folder / "metrics.tmp"
        pd.DataFrame(rows).to_csv(temp, index=False)
        temp.replace(folder / "metrics.csv")
        print(f"{tag} | {len(ids)} training labels", flush=True)
        return head

    for seed in SEEDS:
        start = perf_counter()
        order = np.random.Generator(np.random.PCG64(seed)).permutation(len(pool))
        initial = order[:TRAIN_COUNTS[5]]
        initial_head = evaluate(initial, 5, seed, "Shared", perf_counter() - start)
        for policy in ("Random", "Entropy"):
            selected = initial
            head = initial_head
            for budget in (10, 20):
                count = TRAIN_COUNTS[budget]
                start = perf_counter()
                if policy == "Random":
                    selected = order[:count]
                else:
                    selected = add_uncertain(x, selected, count, head)
                head = evaluate(selected, budget, seed, policy, perf_counter() - start)

    evaluate(np.arange(len(pool)), 100, "all", "Full", 0.0)


def main():
    import torch
    import torchvision
    from torchvision.datasets import CIFAR10
    from gpu_check import check_gpu
    from image_features import extract_features, load_encoder

    check_gpu()
    train = CIFAR10(root=str(ROOT / "data"), train=True, download=False)
    labels = np.asarray(train.targets)
    validation, pool, summary = read_split(labels)
    folder = ROOT / "results" / datetime.now().strftime("validation_baselines_%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    print("Output:", folder, flush=True)
    environment = dict(protocol="1.0", split=summary, seeds=SEEDS, train_counts=TRAIN_COUNTS,
                       models=MODELS, batch_size=BATCH_SIZE, python=platform.python_version(),
                       platform=platform.platform(), torch=torch.__version__,
                       torchvision=torchvision.__version__, numpy=np.__version__,
                       sklearn=sklearn.__version__, gpu=torch.cuda.get_device_name(0))
    (folder / "run_details.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    rows = []
    for name in MODELS:
        print(f"Extracting {name} features...", flush=True)
        model, preprocess = load_encoder(name, "cuda")
        start = perf_counter()
        x = extract_features(model, preprocess, train, pool, "cuda", BATCH_SIZE)
        views = {}
        for domain in ("clean", "noise"):
            views[domain] = extract_features(model, preprocess, train, validation, "cuda",
                                             BATCH_SIZE, noisy=(domain == "noise"))
        environment.setdefault("feature_seconds", {})[name] = perf_counter() - start
        (folder / "run_details.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
        run_encoder(name, pool, x, labels, validation, views, folder, rows)

    metrics = pd.DataFrame(rows)
    paired = metrics[metrics.budget.isin([10, 20])].pivot(
        index=["arch", "budget", "domain", "seed"], columns="policy", values="f1"
    )
    paired["delta_pp"] = 100 * (paired["Entropy"] - paired["Random"])
    paired.to_csv(folder / "paired_by_seed.csv")
    summary = paired.groupby(["arch", "budget", "domain"])["delta_pp"].agg(
        ["count", "mean", "std", "min", "max"]
    )
    summary.to_csv(folder / "paired_summary.csv")
    print(summary.round(3))
    print("Finished:", folder, flush=True)


if __name__ == "__main__":
    main()
