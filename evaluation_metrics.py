import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

def assess(prob, labels):
    pred = prob.argmax(1)
    conf = prob.max(1)
    correct = pred == labels
    bins = np.minimum((10 * conf).astype(int), 9)
    ece = sum(abs((conf[bins == k] - correct[bins == k]).sum()) for k in range(10)) / len(labels)
    return dict(acc=correct.mean(), f1=f1_score(labels, pred, average='macro'),
                ece=ece, brier=np.square(prob - np.eye(prob.shape[1])[labels]).sum(1).mean())


def detection(id_scores, ood_scores):
    # Higher score means OOD; positive examples are held-out classes.
    y = np.r_[np.zeros(len(id_scores)), np.ones(len(ood_scores))]
    score = np.r_[id_scores, ood_scores]
    fpr, tpr, _ = roc_curve(y, score)
    return roc_auc_score(y, score), fpr[tpr >= .95][0]


