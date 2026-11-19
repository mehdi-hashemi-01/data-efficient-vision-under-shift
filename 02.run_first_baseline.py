from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from gpu_check import check_gpu
from cifar10_data import PROJECT, load_data
from image_features import load_encoder, extract_features

MODEL_NAME = "resnet18"
LABEL_PERCENT = 5
BATCH_SIZE = 32
SEED = 11


def run():
    check_gpu()
    if not 0 < LABEL_PERCENT <= 100 or BATCH_SIZE < 1:
        raise ValueError("Choose a percentage from 1 to 100 and a positive batch size.")
    train, test = load_data()
    train_labels = np.array(train.targets)
    pool = np.flatnonzero(train_labels < 8)
    test_ids = np.flatnonzero(np.array(test.targets) < 8)
    test_labels = np.array(test.targets)[test_ids]
    order = np.random.default_rng(SEED).permutation(pool)
    selected = order[:round(len(pool) * LABEL_PERCENT / 100)]
    model, preprocess = load_encoder(MODEL_NAME, "cuda")
    print("Training labels:", len(selected), "| Test images:", len(test_ids))
    print("Extracting training features on the GPU...")
    x_train = extract_features(model, preprocess, train, selected, "cuda", BATCH_SIZE)
    head = LogisticRegression(max_iter=1000)
    head.fit(x_train, train_labels[selected])
    rows = []
    for domain in ["clean", "noisy"]:
        print("Evaluating:", domain)
        x_test = extract_features(model, preprocess, test, test_ids, "cuda", BATCH_SIZE, noisy=(domain == "noisy"))
        predicted = head.predict(x_test)
        rows.append(dict(model=MODEL_NAME, labels=len(selected), percent=LABEL_PERCENT, seed=SEED,
                         domain=domain, accuracy=accuracy_score(test_labels, predicted),
                         macro_f1=f1_score(test_labels, predicted, average="macro")))
    folder = PROJECT / "results" / datetime.now().strftime("baseline_%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    results = pd.DataFrame(rows)
    results.to_csv(folder / "baseline_scores.csv", index=False)
    np.save(folder / "selected_train_ids.npy", selected)
    print(results.to_string(index=False))
    print("Saved in:", folder)


if __name__ == "__main__":
    run()
