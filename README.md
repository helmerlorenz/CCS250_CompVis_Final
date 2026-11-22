# iMAGE ? Image Denoiser

FastAPI website with preview filters and real inference using Restormer, DnCNN, and NAFNet. Three Jupyter notebooks prepare paired data, fine-tune the models, and evaluate/deploy them.

## Course method choice

The team selected **three pretrained denoising models**: Restormer, pretrained DnCNN, and NAFNet. The written `Projects final (1).pdf` describes one classical, one trained-from-scratch, and one pretrained method. The team reports a later verbal clarification, relayed through a classmate, that permits three methods from one category. Keep that clarification as separate evidence; it is not stated in the PDF. The PDF names Restormer and pretrained DnCNN in its denoising menu, while NAFNet is off-menu and may need proposal approval.

The three notebooks are the development workflow. The PDF requests **one executed notebook that runs top-to-bottom in Colab** plus one `.py` module per method for the final submission. Fine-tuning starts from pretrained checkpoints; the written pretrained slot permits either fine-tuning or running the pretrained model.

## Setup (Windows PowerShell)

Use Python 3.11 with the VS Code Python and Jupyter extensions. Open this project folder in VS Code, then run from its root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

For GPU training, install a CUDA-enabled PyTorch build using the [official PyTorch selector](https://pytorch.org/get-started/locally/). Select `.venv` as the notebook kernel. A CPU can run previews and small smoke tests; full Restormer/NAFNet training is computationally expensive.

## Website

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. The default `preview` mode uses three labeled demonstration filters. To run actual models, restart with `IMAGE_MODE=real`:

```powershell
$env:IMAGE_MODE="real"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Real mode retrieves model architecture files and pretrained checkpoints on first use, then caches them under `cache/`. Internet access, disk space, and trust in upstream code are required. Restormer uses its published real-denoising checkpoint, NAFNet uses its published SIDD checkpoint from a verified mirror, and DnCNN uses its published color-blind Gaussian checkpoint. These are different pretraining conditions, which must be disclosed with the results. Local `weights/{model}.pth` checkpoints take precedence for inference.

## Get the paired data

The repository does not include training images. These sources provide the noisy/clean data for notebook 01:

| Dataset | Source | Download size |
|---|---|---|
| SIDD | [Authors' dataset page](https://abdokamel.github.io/sidd/) or the [SIDD archive mirrored by VCLab-PolyU](https://huggingface.co/datasets/VCLab-PolyU/GGT-100K/blob/main/Existing-dataset/SIDD.zip) | The mirror archive is about 13.8 GB. |
| PolyU real-world noisy images | [Authors' repository and download links](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset) or the [PolyU archive mirrored by VCLab-PolyU](https://huggingface.co/datasets/VCLab-PolyU/GGT-100K/blob/main/Existing-dataset/PolyU-Real-World-Noisy-Images-Dataset.zip) | The mirror archive is about 400 MB. |

To download the two verified mirror archives, check their SHA-256 hashes, extract paired images, and build the manifest, run from the project root:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_real_data
```

The transfer resumes completed chunks after interruption. Allow room for both archives and their extracted images. Only these two datasets are used; the entire GGT-100K collection is much larger. If you obtain data from the authors' sources instead, place extracted files under `data/raw/SIDD/` and `data/raw/PolyU/`, inspect the folder layout, then set `SIDD_BASE` and `POLYU_BASE` in [notebook 01](notebooks/01_prepare_5000_images.ipynb) to the paired `GT` and `LQ` directories. Check the source terms before using or redistributing images.

Both archives have been verified and extracted in this workspace. PolyU contains 4,000 noisy/clean pairs across 40 scenes, and SIDD contains 16,000 pairs. The fixed `data/splits.csv` has 5,000 rows; its SHA-256 is `49116dee666c137f25bc76db5196326c5e64db20ce3ae9cd755b3f3e79b689e2`. The split generator assigns whole scenes to train/validation/test, then samples the requested number of crops from each partition. Unused crops are omitted, so exact quotas do not require splitting a scene.

## Paired real-noise experiment

The current experiment uses **5,000 aligned noisy/clean pairs**: 3,500 SIDD and 1,500 PolyU. The fixed quotas are 4,000 training (2,800 SIDD, 1,200 PolyU), 500 validation (350, 150), and 500 test (350, 150). No COCO images or synthetic Gaussian noise are used. Splits are assigned by original scene group to avoid patch leakage.

1. Download and extract the paired data using the sources above, then open [notebook 01](notebooks/01_prepare_5000_images.ipynb). Set the SIDD/PolyU directories to the actual `GT` (clean) and `LQ` (noisy) folders. Verify filename matching, alignment, and scene IDs. The notebook writes `data/inventory/SIDD.csv` and `PolyU.csv`, then uses `scripts/prepare_mixed_dataset.py` to write `data/splits.csv` once. Inventory paths are relative to `data/`. The generator assigns scenes to splits, samples within each split to meet exact quotas, and fails without writing if the inventory cannot support them. Keep the manifest unchanged across all model runs.
2. Open [notebook 02](notebooks/02_finetune_models.ipynb) only if you want an additional fine-tuning experiment. Start in smoke mode, then train with GPU resources if validation suggests an improvement. This workspace completed a 200-step DnCNN CPU trial; its validation PSNR fell, so the published baseline was retained. The trial checkpoint and record are under `HeraldFolder/`.
3. Open the single [submission notebook](HeraldFolder/FINAL_SUBMISSION.ipynb), also mirrored as [notebook 03](notebooks/03_evaluate_and_deploy.ipynb). It checks the saved summaries against the manifest and checkpoint hashes, or computes them if absent. This workspace's final comparison uses **64 px center crops** on all 500 test pairs for each model because CPU-only Restormer evaluation at 128 px was too slow. PSNR/SSIM are patch-based, not full-resolution metrics. Results and a comparison plot are saved under `HeraldFolder/test/`. The older outputs from a different manifest are archived under `HeraldFolder/archive/` and excluded from the current report.

The API recommends a model from the 150 held-out PolyU test pairs only when all three deployed checkpoints match full evaluation summaries under the same manifest and settings. This avoids using SIDD scores affected by published pretraining overlap for the default recommendation. For a single upload with a clean reference, it computes metrics for that upload. Without a clean reference or matching benchmark, it makes no quality claim.

## Project layout

- `app/`: API, paired-image loader, training/evaluation methods, inference, model/cache adapters.
- `methods/`: one runnable Python module per submitted model (`python -m methods.restormer`, `python -m methods.dncnn`, `python -m methods.nafnet`). Use `--input`, `--output`, and optionally `--checkpoint`, or `--evaluate` for the full test split.
- `notebooks/`: one notebook per preparation, fine-tuning, and evaluation step.
- `HeraldFolder/`: measured validation and final test results, trial checkpoints, and submission artifacts for this run.
- `scripts/prepare_real_data.py`: resumable download, extraction, inventory, and split preparation.
- `scripts/prepare_mixed_dataset.py`: paired inventory validation and scene-group split generation.
- `data/raw/`, `data/inventory/`, `data/splits.csv`: local data, excluded from Git.
- `cache/`, `weights/`, `results/`: downloaded sources/weights and local outputs, excluded from Git.
- `tests/`: API, notebook, and dataset workflow checks.

## Limits and provenance

The Git repository does not track datasets, the manifest, or model checkpoints, but this workspace has the verified datasets, manifest, a local Python environment, and the three pretrained checkpoints. Results for Herald's run are saved in `HeraldFolder/`. The current environment has CPU-only PyTorch, so full fine-tuning of Restormer and NAFNet needs a GPU environment. Representative validation pairs were checked for dimensions and pixel alignment.

**Pretraining overlap:** The published real-denoising Restormer and SIDD NAFNet checkpoints were trained on SIDD training data. The project's SIDD subset comes from that same training collection, so SIDD validation/test scores cannot be claimed as independent held-out performance for those pretrained models. Report SIDD and PolyU separately; PolyU provides a more useful cross-dataset check for the published checkpoints. The 500-pair aggregate is a project comparison, with this limitation disclosed.

Online source files come from the [Restormer](https://github.com/swz30/Restormer), [KAIR/DnCNN](https://github.com/cszn/KAIR), and [NAFNet](https://github.com/megvii-research/NAFNet) projects; the NAFNet checkpoint is fetched from a [third-party mirror](https://huggingface.co/mikestealth/nafnet-models). The mirror checkpoint is SHA-256 checked by the loader. Real model downloads and checkpoint compatibility still need verification in the target environment. Report dataset provenance, grouping policy, checkpoint provenance, center-crop size, device, and measured results when publishing comparisons.
