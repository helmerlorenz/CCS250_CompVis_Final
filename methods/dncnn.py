"""Color DnCNN method with a published Gaussian blind denoising checkpoint.

Source: https://github.com/cszn/KAIR
"""
from .common import run_method


if __name__ == "__main__":
    run_method("dncnn")
