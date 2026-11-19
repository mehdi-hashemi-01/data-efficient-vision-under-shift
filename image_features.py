"""Use pretrained CNNs as feature extractors."""
import numpy as np
import torch
from PIL import Image
from torchvision import models


def load_encoder(name, device):
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()
    elif name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier = torch.nn.Identity()
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.to(device).eval(), weights.transforms()


def add_noise(image, index):
    clean = np.asarray(image, dtype=np.float32) / 255
    rng = np.random.default_rng(900 + int(index))
    noisy = np.clip(clean + rng.normal(0, 0.12, clean.shape), 0, 1)
    return Image.fromarray((noisy * 255).round().astype("uint8"))


@torch.inference_mode()
def extract_features(model, preprocess, dataset, indices, device, batch_size, noisy=False):
    features = []
    for start in range(0, len(indices), batch_size):
        batch = []
        for index in indices[start:start + batch_size]:
            image, _ = dataset[int(index)]
            if noisy:
                image = add_noise(image, index)
            batch.append(preprocess(image))
        tensor = torch.stack(batch).to(device)
        features.append(model(tensor).cpu().numpy())
    return np.concatenate(features)
