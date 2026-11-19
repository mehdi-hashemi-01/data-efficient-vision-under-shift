"""Run this file in PyCharm before downloading the dataset."""
import sys


def check_gpu():
    print("Python interpreter:", sys.executable)
    import torch
    print("PyTorch:", torch.__version__, "| Built with CUDA:", torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check this interpreter, the CUDA build of PyTorch, and the NVIDIA driver.")
    import torchvision
    print("Torchvision:", torchvision.__version__)
    print("GPU:", torch.cuda.get_device_name(0))
    value = torch.ones(2, device="cuda").sum().item()
    print("GPU calculation:", value, "(expected 2.0)")


