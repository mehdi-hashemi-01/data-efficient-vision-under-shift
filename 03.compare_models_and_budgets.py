import json
import platform
import warnings
from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torchvision
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from cifar10_data import PROJECT, load_data
from evaluation_metrics import assess, detection
from gpu_check import check_gpu
from image_features import extract_features, load_encoder

MODEL_NAMES = ["resnet18", "mobilenet_v3_small"]
SEEDS = [11, 22, 33, 44, 55]
BUDGETS = [5, 10, 20, 100]
BATCH_SIZE = 32
MAX_ITER = 3000


def select_images(policy, order, selected, count, features, head):
    # Both policies start with the same labeled images.
    if head is None or policy == "Random":
        return order[:count]

    remaining = np.setdiff1d(order, selected)
    probabilities = head.predict_proba(features[remaining])
    log_probabilities = np.log(probabilities.clip(min=1e-15))
    entropy = -(probabilities * log_probabilities).sum(axis=1)

    ranked = np.argsort(-entropy)
    needed = count - len(selected)
    new_images = remaining[ranked[:needed]]
    return np.concatenate([selected, new_images])


def train_head(features, labels):
    for solver in ["lbfgs", "newton-cg"]:
        head = LogisticRegression(C=1, solver=solver, max_iter=MAX_ITER)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                head.fit(features, labels)
            return head
        except ConvergenceWarning as warning:
            print(f"{solver}: {warning}", flush=True)
            if solver == "newton-cg":
                raise RuntimeError("Neither solver converged on this training subset.") from warning
            print("Retrying with newton-cg.", flush=True)


def ood_scores(head, known, unknown):
    known_msp = 1 - head.predict_proba(known).max(axis=1)
    unknown_msp = 1 - head.predict_proba(unknown).max(axis=1)
    known_energy = -logsumexp(head.decision_function(known), axis=1)
    unknown_energy = -logsumexp(head.decision_function(unknown), axis=1)
    msp_auc, msp_fpr = detection(known_msp, unknown_msp)
    energy_auc, energy_fpr = detection(known_energy, unknown_energy)
    return dict(msp_auc=msp_auc, msp_fpr95=msp_fpr,
                energy_auc=energy_auc, energy_fpr95=energy_fpr)


@torch.inference_mode()
def measure_gpu(model, head, image):
    # Include the fitted classifier in the GPU timing.
    weight = torch.as_tensor(head.coef_, device=image.device, dtype=image.dtype)
    bias = torch.as_tensor(head.intercept_, device=image.device, dtype=image.dtype)
    times = []
    for step in range(60):
        if step == 10:
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = perf_counter()
        features = model(image)
        torch.nn.functional.linear(features, weight, bias).softmax(dim=1)
        torch.cuda.synchronize()
        if step >= 10:
            times.append(1000 * (perf_counter() - start))
        del features

    tensors = list(model.parameters()) + list(model.buffers()) + [weight, bias]
    size_bytes = sum(t.numel() * t.element_size() for t in tensors)
    return dict(median_ms=np.median(times), weights_buffers_mib=size_bytes / 2**20,
                gpu_peak_mib=torch.cuda.max_memory_allocated() / 2**20)


def run():
    check_gpu()
    train, test = load_data()
    train_labels = np.array(train.targets)
    test_labels = np.array(test.targets)
    pool = np.flatnonzero(train_labels < 8)
    known = np.flatnonzero(test_labels < 8)
    # Ships and trucks are held out for OOD evaluation.
    unknown = np.flatnonzero(test_labels >= 8)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = PROJECT / "results" / f"comparison_{stamp}"
    folder.mkdir(parents=True)
    print("Output:", folder, flush=True)
    environment = dict(torch=torch.__version__, torchvision=torchvision.__version__, device="cuda",
                       gpu=torch.cuda.get_device_name(0), platform=platform.platform(),
                       seeds=SEEDS, batch_size=BATCH_SIZE, max_iter=MAX_ITER)
    (folder / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    rows = []
    for name in MODEL_NAMES:
        print(f"Extracting features: {name}", flush=True)
        model, preprocess = load_encoder(name, "cuda")
        train_x = extract_features(model, preprocess, train, pool, "cuda", BATCH_SIZE)
        clean_x = extract_features(model, preprocess, test, known, "cuda", BATCH_SIZE)
        noisy_x = extract_features(model, preprocess, test, known, "cuda", BATCH_SIZE, noisy=True)
        unknown_x = extract_features(model, preprocess, test, unknown, "cuda", BATCH_SIZE)
        first_image, _ = test[int(known[0])]
        one_image = preprocess(first_image).unsqueeze(0).to("cuda")

        for seed in SEEDS:
            order = np.random.default_rng(seed).permutation(len(pool))
            for policy in ["Random", "Entropy"]:
                head = None
                selected = np.array([], dtype=int)
                for percent in BUDGETS:
                    count = round(len(pool) * percent / 100)
                    print(f"{name} | seed {seed} | {policy} | {percent}% | {count} labels", flush=True)
                    selected = select_images(policy, order, selected, count, train_x, head)
                    train_ids = pool[selected]
                    head = train_head(train_x[selected], train_labels[train_ids])
                    np.save(folder / f"{name}_{seed}_{policy}_{percent}_ids.npy", train_ids)

                    info = dict(arch=name, seed=seed, policy=policy, budget=percent, n=count,
                                solver=head.solver, iterations=int(head.n_iter_.max()))
                    info.update(ood_scores(head, clean_x, unknown_x))
                    info.update(measure_gpu(model, head, one_image))
                    for domain, features in [("clean", clean_x), ("noise", noisy_x)]:
                        probabilities = head.predict_proba(features)
                        scores = assess(probabilities, test_labels[known])
                        rows.append({**info, "domain": domain, **scores})

                    temporary = folder / "metrics.tmp"
                    pd.DataFrame(rows).to_csv(temporary, index=False)
                    temporary.replace(folder / "metrics.csv")
    print("Finished:", folder / "metrics.csv")


if __name__ == "__main__":
    run()
