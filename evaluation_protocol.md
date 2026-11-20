# Evaluation protocol for label-efficient CIFAR-10 classification

Version 1.0 · Prospective protocol for experiments after script 08

## Research question and primary comparison

With a fixed annotation budget, does the proposed acquisition rule improve recognition of known classes under Gaussian noise compared with predictive entropy sampling?

The primary comparison uses frozen ResNet18 features, a 20% total label budget and macro-F1 on the noisy known-class test images at Gaussian noise standard deviation 0.12. For each seed, compute 100 × (proposed macro-F1 − entropy macro-F1). The primary result is the arithmetic mean of these five paired differences in percentage points. A positive value favors the proposed rule. Report the result regardless of its sign. Record the proposed acquisition formula and its candidate settings in a dated commit before running its comparison.

This choice follows exploratory work in scripts 02–08, including inspection of the official test predictions. Results on this test set therefore extend the exploratory benchmark. An independent generalization claim requires an untouched evaluation dataset with a compatible label mapping, chosen before examining its results.

## Data partitions and image IDs

Use the original zero-based row index within each torchvision CIFAR-10 split as the image ID. Store the split name with it. The simulated annotation pool contains the 40,000 training images with class IDs 0–7. This benchmark assumes the known-class pool has already been identified. Ship and truck training images are excluded.

Create the validation partition once, before feature fitting or label acquisition. The following rule picks 400 images uniformly without consulting their individual class labels after the known-class pool has been formed.

```python
known_ids = np.flatnonzero(np.isin(train_targets, np.arange(8)))
split_rng = np.random.Generator(np.random.PCG64(20261120))
shuffled_ids = split_rng.permutation(known_ids)
validation_ids = np.sort(shuffled_ids[:400])
pool_ids = np.sort(shuffled_ids[400:])
```

Save both arrays and their SHA-256 file hashes in the first run. Reuse them across seeds, encoders, methods and budgets. Verify that the two arrays are disjoint and together contain all 40,000 known training IDs. Record validation class counts after splitting. Keep the original draw. Validation images are excluded from acquisition, classifier fitting and any local self-supervised training.

Use the 8,000 official known-class test images for classification and the 2,000 official ship/truck test images for semantic OOD evaluation. For the paired noise condition, reuse the audited image function from script 05 with standard deviation 0.12 and seed 900 + original image ID. Apply noise to 32 × 32 RGB before encoder preprocessing. Generate a corresponding noisy validation view by the same rule using the training image ID. Each view inherits the original label.

## Annotation budget

The denominator is the original 40,000 known training images. Count each unique validation label once in every method's budget. Reusing the same label across rounds or views incurs no additional charge.

| Total budget | Total labels | Fixed validation labels | Cumulative classifier training labels | New training labels this round |
|---|---:|---:|---:|---:|
| 5% | 2,000 | 400 | 1,600 | 1,600 |
| 10% | 4,000 | 400 | 3,600 | 2,000 |
| 20% | 8,000 | 400 | 7,600 | 4,000 |
| 100% | 40,000 | 400 | 39,600 | 32,000 |

At each budget, verify the selected IDs are unique, belong to the acquisition pool and include all IDs selected in earlier rounds. Repeat the reference baselines under this protocol. For example, the earlier 20% run trained on 8,000 labels, while this protocol trains on 7,600 and reserves 400 for validation.

## Training, acquisition and model selection

1. Use acquisition seeds 11, 22, 33, 44 and 55. For each seed, permute the sorted pool IDs with NumPy PCG64. Share the first 1,600 IDs across all methods and both encoders. Random extends this same permutation at each round.
2. For Entropy, score each remaining image by the predictive entropy of the previous round's classifier on its clean view. Acquire the highest scores in one batch. Break exact ties by ascending original image ID. The proposed rule follows the same acquisition schedule. Reveal the simulated labels only after selecting IDs.
3. Compare selection rules with the same frozen ImageNet-1K V1 weights, preprocessing and logistic regression settings from script 03 (C = 1). Refit the classifier on all acquired labels after each batch. Fit any learned feature transformation using only the acquired training examples. Record solver, iterations and convergence warnings. At 100%, use the same sorted training IDs for all methods and fit one common full-label reference per encoder.
4. Before tuning a new rule, record its finite candidate list, default setting and deterministic tie order. Select one setting by mean noisy-validation macro-F1 across the five seeds at 20% with ResNet18. Resolve a tie by the recorded order. Freeze that setting across secondary budgets and encoders. Record all candidate validation scores. Use only the validation labels for tuning, including any later temperature calibration.
5. Freeze the code revision, configuration and selected setting before scoring the official test data. Changes made after seeing those scores start a separately identified exploratory run.

## Secondary outcomes and reporting

| Question | Measurement |
|---|---|
| Does performance transfer to clean images or another encoder? | Clean and noisy macro-F1 and accuracy for ResNet18 and MobileNetV3-Small at each budget. Compute macro-F1 over fixed labels 0–7 with undefined class F1 set to zero. Include Random and Entropy. |
| Which classes account for a change? | Per-class F1, confusion matrices and paired clean-to-noisy correctness transitions using original test IDs. |
| Are the predicted probabilities reliable? | Report top-label ECE with the existing 10 equal-width bins and the multiclass Brier score separately for clean and noisy known images. |
| Can the model detect the two held-out classes? | Compare clean known images against ship/truck using MSP and energy AUROC and FPR@95 as implemented in script 03. Unknown is positive. Keep this task separate from known-class recognition under noise. |
| What does selection cost? | Record acquisition time per round separately from feature extraction, head fitting and inference. Report hardware, batch size, timing procedure and peak GPU memory. |

Save per-seed scores, paired differences, sample standard deviation, range and win/tie/loss counts. Show all five differences for the primary comparison. The five seeds measure variability from acquisition on a fixed split. They share test images and a noise realization. Do not interpret the resulting spread as uncertainty across independent test datasets.

## Records required for a run

Save the protocol version and Git revision, environment, partition arrays and hashes, class counts, acquisition IDs per round, validation settings and scores, final test predictions and metric tables. Check that every method has the same five seeds and test IDs before pairing. If a run fails, record the cause and rerun the same configuration. Mark missing pairs explicitly and withhold the five-seed primary mean until all five pairs are available.
