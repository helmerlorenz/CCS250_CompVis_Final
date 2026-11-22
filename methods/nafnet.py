"""NAFNet width-32 SIDD denoising method using the authors' architecture.

Source: https://github.com/megvii-research/NAFNet
"""
from .common import run_method


if __name__ == "__main__":
    run_method("nafnet")
