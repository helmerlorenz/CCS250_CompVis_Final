# Herald's project results

This folder holds results generated in this workspace. `validation/` contains
reproducible validation metrics from `python -m scripts.benchmark_validation`.
`test/` contains the three full 500-pair test summaries and per-image CSV files
from identical 64 px center crops. `FINAL_SUBMISSION.ipynb` was executed locally
from top to bottom; `iMAGE_report.pdf` and `iMAGE_presentation.pptx` summarize
the measurements. `CONTRIBUTIONS.md` awaits the group details.

The dataset stays under `../data/`; pretrained checkpoints stay under
`../cache/pretrained/`. Large data and checkpoint files are not duplicated here.
Historical outputs from the prior version of notebook 03 are preserved under
`archive/`; they refer to a different manifest and are not current results.

The instructor approved NAFNet according to the project owner. Group number
and member names are still pending. The notebook has not yet been run in Colab;
that requires the full project and untracked dataset to be available there.
