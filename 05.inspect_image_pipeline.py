from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torchvision.models import ResNet18_Weights

from cifar10_data import PROJECT, load_data
from image_features import add_noise

NOISE_SIGMA = 0.12
CLASS_COUNT = 8
EXAMPLE_SEED = 11


def trace_noise(image, index, sigma):
    # Recreate the pre-clipping values that add_noise does not return.
    clean = np.asarray(image, dtype=np.float32) / 255
    rng = np.random.default_rng(900 + int(index))
    raw = clean + rng.normal(0, sigma, clean.shape)
    clipped = np.count_nonzero((raw < 0) | (raw > 1))
    pixels = (np.clip(raw, 0, 1) * 255).round().astype("uint8")
    return Image.fromarray(pixels), clipped


def pixel_difference(first, second):
    a = np.asarray(first, dtype=np.int16)
    b = np.asarray(second, dtype=np.int16)
    return np.abs(a - b)


def display_tensor(tensor, preprocess):
    # Undo normalization only for the preview.
    mean = torch.tensor(preprocess.mean).view(3, 1, 1)
    std = torch.tensor(preprocess.std).view(3, 1, 1)
    rgb = (tensor * std + mean).clamp(0, 1)
    pixels = (rgb.permute(1, 2, 0) * 255).round().byte().numpy()
    return Image.fromarray(pixels)


def save_examples(test, examples, preprocess, folder):
    width, height = 224, 260
    sheet = Image.new("RGB", (4 * width, 36 + len(examples) * height), "white")
    draw = ImageDraw.Draw(sheet)
    headings = ["Clean | 32 x 32 (enlarged)", "Noise | 32 x 32 (enlarged)",
                "Clean | network input", "Noise | network input"]
    for column, title in enumerate(headings):
        draw.text((column * width + 6, 10), title, fill="black")

    for row, index in enumerate(examples):
        clean, label = test[int(index)]
        noisy = add_noise(clean, index)
        panels = [clean.resize((width, width), Image.Resampling.NEAREST),
                  noisy.resize((width, width), Image.Resampling.NEAREST),
                  display_tensor(preprocess(clean), preprocess),
                  display_tensor(preprocess(noisy), preprocess)]
        top = 36 + row * height
        for column, panel in enumerate(panels):
            sheet.paste(panel, (column * width, top))
        draw.text((6, top + width + 5), f"{test.classes[label]}  |  test ID {index}", fill="black")
    sheet.save(folder / "clean_vs_noisy.png")


def run():
    _, test = load_data()
    labels = np.array(test.targets)
    indices = np.flatnonzero(labels < CLASS_COUNT)
    preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
    torch.set_num_threads(1)
    folder = PROJECT / "results" / datetime.now().strftime("pipeline_%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    print("Output:", folder, flush=True)

    rows = []
    for number, index in enumerate(indices, start=1):
        image, label = test[int(index)]
        noisy = add_noise(image, index)
        traced, clipped = trace_noise(image, index, NOISE_SIGMA)
        zero, _ = trace_noise(image, index, 0.0)

        delta = pixel_difference(image, noisy)
        zero_pixel_diff = int(pixel_difference(image, zero).max())
        zero_tensor_diff = float((preprocess(image) - preprocess(zero)).abs().max())
        trace_diff = int(pixel_difference(noisy, traced).max())
        rows.append({
            "test_id": index, "class": test.classes[label], "rgb_values": delta.size,
            "absolute_change_sum_uint8": int(delta.sum()), "clipped_rgb_values": clipped,
            "zero_pixel_max_uint8": zero_pixel_diff, "zero_tensor_max_abs": zero_tensor_diff,
            "trace_vs_project_max_uint8": trace_diff,
        })
        if number % 1000 == 0:
            print(f"Checked {number}/{len(indices)}", flush=True)

    data = pd.DataFrame(rows)
    data.to_csv(folder / "per_image.csv", index=False)
    total = int(data.rgb_values.sum())
    summary = {
        "test_images": len(data), "noise_sigma": NOISE_SIGMA,
        "mean_abs_pixel_change_0_to_1": data.absolute_change_sum_uint8.sum() / (255 * total),
        "clipped_rgb_percent": 100 * data.clipped_rgb_values.sum() / total,
        "zero_pixel_max_uint8": data.zero_pixel_max_uint8.max(),
        "zero_tensor_max_abs": data.zero_tensor_max_abs.max(),
        "trace_vs_project_max_uint8": data.trace_vs_project_max_uint8.max(),
    }
    passed = (summary["zero_pixel_max_uint8"] == 0 and summary["zero_tensor_max_abs"] == 0
              and summary["trace_vs_project_max_uint8"] == 0)
    summary["status"] = "PASS" if passed else "FAIL"
    pd.DataFrame([summary]).to_csv(folder / "pipeline_summary.csv", index=False)
    rng = np.random.default_rng(EXAMPLE_SEED)
    examples = [rng.choice(np.flatnonzero(labels == label)) for label in range(CLASS_COUNT)]
    save_examples(test, examples, preprocess, folder)
    print(pd.Series(summary).to_string())
    if not passed:
        raise RuntimeError("Pipeline check failed. Inspect the nonzero difference columns in per_image.csv.")
    print("Saved:", folder / "clean_vs_noisy.png")


if __name__ == "__main__":
    run()
