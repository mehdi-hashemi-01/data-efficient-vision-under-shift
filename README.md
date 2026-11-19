<div align="center">

# Data-Efficient Vision Under Shift

**How far can 2,000 labels take a pretrained image classifier when its inputs change?**

![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-176B69)
![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C)
![Stage](https://img.shields.io/badge/Stage-Five--seed_benchmark-475569)

[Overview](#overview) · [Dataset](#dataset-and-label-budgets) · [First baseline](#first-baseline) · [Results](#benchmark-results) · [Reproduce](#reproduce-in-pycharm) · [Roadmap](#research-roadmap)

</div>

## Overview

Imagine a pool of 40,000 images and a budget to label only 2,000. A frozen image encoder can already describe each image with a vector of features. How accurately can a classifier learn eight categories from those vectors? If a clean image of a cat becomes noisy, does the prediction change? And when the input is a ship, can the classifier signal that none of its eight answers fits?

This repository studies these questions with frozen **ResNet18** and **MobileNetV3-Small** encoders. The first experiment measures how the same classifier performs on clean and noisy versions of the same images. The second compares which images to label next and evaluates confidence, unfamiliar-class detection and inference cost.

| Stage | Research question | Evidence available |
|---|---|---|
| First baseline | How well does ResNet18 transfer with 5% of the training labels? | Measured on 8,000 paired test images. |
| Model and budget comparison | At equal label budgets, does selecting uncertain images improve on random sampling? | Completed across four budgets and five seeds. |
| Representation and selection controls | How much do pretraining, sample diversity and prediction stability each contribute? | Planned experiments. |

## Dataset and label budgets

CIFAR-10 contains 60,000 RGB images of 32 × 32 pixels. The original split contains 50,000 training images and 10,000 test images. This project defines an eight-class downstream task using the original split.

| Partition | Classes | Images | Use |
|---|---|---:|---|
| Known training pool | Airplane, automobile, bird, cat, deer, dog, frog, horse | 40,000 | Simulate purchasing labels and fit the classifier |
| Known clean test | Same eight classes | 8,000 | Measure classification and confidence |
| Known noisy test | Same eight classes and image IDs | 8,000 paired views | Measure sensitivity to Gaussian noise |
| Unknown test | Ship and truck | 2,000 | Evaluate semantic out-of-distribution (OOD) detection |

The 10,000 training photographs of ships and trucks are excluded. The noisy test set reuses the clean test photographs. Noise is added at the original resolution with standard deviation 0.12 on a 0–1 pixel scale, then clipped to that range. A seed derived from the image index gives every model the same noisy view.

| Label budget | 5% | 10% | 20% | 100% |
|---|---:|---:|---:|---:|
| Known training images with revealed labels | 2,000 | 4,000 | 8,000 | 40,000 |

Dataset source and class definitions: [CIFAR-10, University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html).

## First baseline

1. Randomly select 2,000 known-class training images using seed 11.
2. Load the ImageNet-1K pretrained ResNet18, remove its final classification layer, and apply the preprocessing supplied with the weights.
3. Extract features on the GPU with the encoder frozen. Fit a logistic regression classifier on the CPU using the selected labels.
4. Measure accuracy and macro-F1 twice, first on clean test images and then on their noisy counterparts. Save the scores and selected training IDs.

## Benchmark results

### First run · ResNet18 · 2,000 labels · seed 11

| Test condition | Images | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Clean | 8,000 | 0.786500 | 0.786832 |
| Gaussian noise | 8,000 | 0.220125 | 0.146720 |

Run `baseline_20261119_220732_930634`. Adding noise reduced macro-F1 by 64.01 percentage points and accuracy by 56.64 points. These figures come from script 02. The corresponding fit in script 03 used the same selected IDs and produced slightly different scores.

### Model and budget comparison

Both acquisition policies start with the same randomly selected 5% for a given seed. At the next budget, **Random** continues along a fixed random ordering. **Entropy** ranks the remaining images by uncertainty under the previous classifier, reveals labels for the highest-ranked images and refits the head. At 100%, both policies have access to the entire pool. The intermediate 10% and 20% budgets reveal where the acquisition policy makes a difference.

Values at 20% labels are means ± sample standard deviations across seeds 11, 22, 33, 44 and 55. Run `comparison_20261119_225145_293414` contains all 80 fitted configurations and 160 domain rows. Every recorded fit used lbfgs (211–1,654 iterations).

| Encoder | Selection | Clean macro-F1 ↑ | Noisy macro-F1 ↑ | Noisy Brier ↓ | MSP AUROC ↑ |
|---|---|---:|---:|---:|---:|
| ResNet18 | Random | 0.8138 ± 0.0035 | 0.1228 ± 0.0259 | 1.3728 ± 0.1197 | 0.5574 ± 0.0242 |
| ResNet18 | Entropy | 0.8415 ± 0.0031 | 0.1073 ± 0.0106 | 1.3771 ± 0.0369 | 0.5176 ± 0.0150 |
| MobileNetV3-Small | Random | 0.8539 ± 0.0048 | 0.2486 ± 0.0152 | 1.0404 ± 0.0341 | 0.5686 ± 0.0266 |
| MobileNetV3-Small | Entropy | 0.8746 ± 0.0016 | 0.2679 ± 0.0135 | 1.0182 ± 0.0251 | 0.5723 ± 0.0076 |

#### Effect of entropy selection

Mean paired macro-F1 difference (Entropy minus Random), in percentage points. Positive values favor Entropy.

| Encoder | Labels | Clean | Gaussian noise |
|---|---:|---:|---:|
| ResNet18 | 10% | +1.24 | -2.74 |
| ResNet18 | 20% | +2.77 | -1.55 |
| MobileNetV3-Small | 10% | +0.76 | +0.50 |
| MobileNetV3-Small | 20% | +2.07 | +1.93 |

The two encoders respond differently under this corruption. At 20%, entropy selection improves clean macro-F1 for both. On noisy images it improves MobileNetV3-Small and reduces ResNet18 macro-F1. Per-image predictions are needed to identify which classes account for the gap.

<details>
<summary><b>How to read the comparison CSV</b></summary>

| Columns in metrics.csv | Interpretation |
|---|---|
| acc, f1 | Accuracy and macro-F1 on the indicated domain. Macro-F1 gives each known class equal weight. |
| ece, brier | Top-label calibration error in 10 equal-width confidence bins, and multiclass Brier score summed across classes. Lower values are better. |
| msp_auc, msp_fpr95 | OOD detection using one minus the largest predicted class probability. An unknown image is the positive class. FPR95 is the fraction of known images flagged as unknown at the first ROC point with at least 95% unknown detection. |
| energy_auc, energy_fpr95 | The same detection task scored using negative log-sum-exp of the head logits. |
| median_ms | Median synchronized forward time on one preprocessed image, including the GPU head, over 50 calls after 10 warm-up calls. |
| weights_buffers_mib, gpu_peak_mib | Parameter and buffer storage, and peak allocated GPU memory during the timed inference section. |

For the comparison script the noisy domain is named `noise`; the first baseline uses `noisy`. OOD metrics compare **clean known** with **unknown** images and repeat on both domain rows. Use one row per seed/configuration when summarizing OOD and runtime.

</details>

## Reproduce in PyCharm

Open the repository folder in PyCharm and choose a Python interpreter with CUDA-enabled PyTorch and compatible torchvision. The remaining packages are listed in [requirements.txt](requirements.txt). The recorded comparison ran on an NVIDIA GeForce RTX 5070 Laptop GPU with CUDA 12.8. Dataset and pretrained-weight downloads require internet access on first use.

Right-click the desired file in the Project panel and choose **Run**. Experiment settings are ordinary Python variables near the top of each experiment file.

| File | Purpose | Output |
|---|---|---|
| 00.check_gpu.py | Check the interpreter, CUDA and GPU computation | GPU name and test calculation in the Run window |
| 01.explore_cifar10.py | Inspect the dataset and display one example per class | cifar10_examples.png and class counts |
| 02.run_first_baseline.py | Run the first clean/noisy comparison | results/baseline_&lt;timestamp&gt;/baseline_scores.csv and selected_train_ids.npy |
| 03.compare_models_and_budgets.py | Compare two encoders, two selection policies and four budgets | results/comparison_&lt;timestamp&gt;/metrics.csv, training-ID files and environment.json |

Script 02 loads the dataset itself, so it can run directly. For a controlled second run, change only `LABEL_PERCENT` from 5 to 10 while keeping `MODEL_NAME` and `SEED` fixed. Output folders are timestamped.

### Repository layout

| Path | Role |
|---|---|
| data/ | Local CIFAR-10 download shared by the experiment scripts |
| cifar10_data.py | Dataset paths and train/test loading |
| gpu_check.py | Shared CUDA check |
| image_features.py | Encoder loading, feature extraction and deterministic image corruption |
| evaluation_metrics.py | Classification, calibration and OOD metrics |
| results/ | Generated run folders |
| README_FA.md | Persian guide to setup, scripts and interpretation |

## Limitations and error analysis

- **Scope of shift.** A single Gaussian noise level measures sensitivity to one corruption. Performance on blur, compression or naturally shifted photographs remains open.
- **Meaning of unknown.** Ships and trucks are held out of the downstream task. Related categories may have been seen during ImageNet pretraining.
- **Seed variation.** All seeds share the same test images and noisy views. Their variation describes sensitivity to the training-label selection on this split.
- **Model selection.** The scripts score the official test partition. Use a training-derived validation split for later choices about augmentations, confidence thresholds and selection rules.
- **Next error audit.** Save image ID, true class, predicted class and confidence for both views. Count images that change from correct to incorrect and from incorrect to correct. This explains what the aggregate performance drop hides. The current baseline saves aggregate scores and selected training IDs.

## Research roadmap

- [x] Record the two rows from the first baseline and their run identifier.
- [x] Compare random and entropy selection at 10% and 20% using paired seed-level differences.
- [ ] Add clean/noisy error examples and per-class changes from saved predictions.
- [ ] Compare frozen supervised features against a matched self-supervised encoder under the same annotation budget.
- [ ] Test whether agreement across two mild image transformations helps select useful uncertain examples. Include diversity-only and uncertainty-plus-diversity controls such as BADGE.
- [ ] Evaluate additional corruptions and an external OOD dataset.

<!-- For each new stage, add the question, the one change being tested, the fixed comparison conditions, the run artifact, measured results, and one limitation. Link figures once their files are committed. -->

## References

- Krizhevsky, Nair and Hinton. [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
- Ash et al. [Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds](https://arxiv.org/abs/1906.03671). ICLR 2020.
- Guo et al. [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). ICML 2017.
- Liu et al. [Energy-based Out-of-distribution Detection](https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html). NeurIPS 2020.

## Maintainer

[Mehdi Hashemi](https://github.com/mehdi-hashemi-01)
