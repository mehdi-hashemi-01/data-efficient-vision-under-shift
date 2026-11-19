import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

from cifar10_data import PROJECT, load_data
from evaluation_metrics import assess, detection
from gpu_check import check_gpu
from image_features import extract_features, load_encoder

SOURCE = PROJECT / "results" / "comparison_20261119_225145_293414"
IDS_FILE = "resnet18_11_Random_20_ids.npy"
IDS_SHA256 = "1a75dbbaa69f2780908da1674de96be2822bd347b4959ebe1a41cce27c0c70bd"


def load_reference():
    path = SOURCE / IDS_FILE
    if hashlib.sha256(path.read_bytes()).hexdigest() != IDS_SHA256:
        raise ValueError(f"Training IDs differ from the archived run: {path}")
    ids = np.load(path, allow_pickle=False)
    scores = pd.read_csv(SOURCE / "metrics.csv")
    scores = scores.query("arch == 'resnet18' and seed == 11 and policy == 'Random' and budget == 20")
    if sorted(scores.domain) != ["clean", "noise"] or not (scores.n == len(ids)).all():
        raise ValueError("Expected two reference rows with the archived label count.")
    settings = json.loads((SOURCE / "environment.json").read_text(encoding="utf-8"))
    return ids, scores.set_index("domain"), settings


def compare(previous, current):
    rows = []
    for domain, scores in current.items():
        ref = previous.loc["clean" if domain == "ood" else domain]
        for metric, value in scores.items():
            old = ref[metric]
            rows.append({"domain": domain, "metric": metric, "previous": old,
                         "replay": value, "difference": value - old})
    return pd.DataFrame(rows)


def run():
    ids, previous, settings = load_reference()
    check_gpu()
    train, test = load_data()
    labels = np.array(train.targets)
    test_labels = np.array(test.targets)
    pool = np.flatnonzero(labels < 8)
    known = np.flatnonzero(test_labels < 8)
    unknown = np.flatnonzero(test_labels >= 8)
    positions = np.searchsorted(pool, ids)
    if not np.array_equal(pool[positions], ids):
        raise ValueError("A saved ID falls outside the known-class pool.")

    model, preprocess = load_encoder("resnet18", "cuda")
    batch = settings["batch_size"]
    # Keep the original batch composition before selecting the saved training rows.
    print("Extracting training features...", flush=True)
    features = extract_features(model, preprocess, train, pool, "cuda", batch)
    head = LogisticRegression(C=1, solver=previous.loc["clean", "solver"],
                              max_iter=settings["max_iter"])
    head.fit(features[positions], labels[ids])
    del features

    current = {}
    print("Evaluating clean images...", flush=True)
    clean = extract_features(model, preprocess, test, known, "cuda", batch)
    clean_prob = head.predict_proba(clean)
    current["clean"] = assess(clean_prob, test_labels[known])

    print("Evaluating noisy images...", flush=True)
    noisy = extract_features(model, preprocess, test, known, "cuda", batch, noisy=True)
    current["noise"] = assess(head.predict_proba(noisy), test_labels[known])

    print("Evaluating held-out classes...", flush=True)
    ood = extract_features(model, preprocess, test, unknown, "cuda", batch)
    ood_prob = head.predict_proba(ood)
    msp_auc, msp_fpr = detection(1 - clean_prob.max(axis=1), 1 - ood_prob.max(axis=1))
    energy_clean = -logsumexp(head.decision_function(clean), axis=1)
    energy_ood = -logsumexp(head.decision_function(ood), axis=1)
    energy_auc, energy_fpr = detection(energy_clean, energy_ood)
    current["ood"] = dict(msp_auc=msp_auc, msp_fpr95=msp_fpr,
                          energy_auc=energy_auc, energy_fpr95=energy_fpr)

    table = compare(previous, current)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = PROJECT / "results" / f"replay_resnet18_{stamp}"
    folder.mkdir(parents=True)
    table.to_csv(folder / "comparison.csv", index=False)
    sources = [Path(__file__), PROJECT / "image_features.py", PROJECT / "evaluation_metrics.py",
               PROJECT / "cifar10_data.py"]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    record = dict(source_run=SOURCE.name, model="resnet18", weights="IMAGENET1K_V1",
                  policy="Random", seed=11, budget=20, n=len(ids), ids_sha256=IDS_SHA256,
                  batch_size=batch, solver=head.solver, max_iter=head.max_iter,
                  original_iterations=int(previous.loc["clean", "iterations"]),
                  replay_iterations=int(head.n_iter_.max()), source_environment=settings,
                  python=platform.python_version(), torch=torch.__version__,
                  torchvision=torchvision.__version__, sklearn=sklearn.__version__,
                  numpy=np.__version__, gpu=torch.cuda.get_device_name(0), code_sha256=hashes)
    (folder / "replay_details.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print("Saved in:", folder)


if __name__ == "__main__":
    run()
