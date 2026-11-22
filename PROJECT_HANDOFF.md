# iMAGE project handoff

Repository: `D:\CodingRelated\Codes.Ams\ComVisFinal\CCS250_CompVis_Final`

## Goal and approved method choice

Compare **three pretrained denoising models** - DnCNN, Restormer, and NAFNet - on the same paired real-noise SIDD/PolyU split. The team chose the three-pretrained option. Do not describe this as COCO, synthetic Gaussian, or training from scratch. Fine-tuning, if done, starts from published checkpoints and is additional to the pretrained baseline comparison.

The written assignment PDF, `Projects final (1).pdf`, describes one classical, one trained-from-scratch, and one pretrained method. **The user reports a later verbal clarification, relayed by a classmate via Messenger, allowing three classical, three scratch-trained, or three pretrained models.** The three-pretrained allowance is a reported clarification, not text found in the PDF; retain the Messenger evidence for the instructor. The written PDF also says off-menu methods need instructor approval: its denoising menu names Restormer and pretrained DnCNN, but not NAFNet. The project owner has confirmed NAFNet was approved; retain that evidence as well. Group number and member names are pending.

Other written deliverables still matter: one executed notebook that runs top-to-bottom in Colab, one `.py` module per method, a dataset link, report PDF, slides, a contribution statement, and disclosure of AI-assistant use. One submission notebook has been created and executed locally; Colab execution remains to be verified after the full project and data are uploaded there.

## Current state (November 22, 2026)

- The data preparation pipeline is in `scripts/prepare_real_data.py`. It downloads verified archives, extracts paired `GT`/`LQ` images, builds inventories, and creates `data/splits.csv` without overwriting an existing manifest. Large files under `data/` are Git-ignored.
- PolyU is complete: 4,000 noisy/clean pairs across 40 scenes are extracted under `data/raw/PolyU/`; `data/inventory/PolyU.csv` exists.
- SIDD's 13.8 GB archive was checksum-verified and extracted under `data/raw/SIDD/SIDD/train_sub512/` (16,000 pairs). `data/inventory/SIDD.csv` exists. The background preparation job finished successfully; see `data/downloads/prepare.log`.
- `data/splits.csv` is complete: 5,000 paired rows; SHA-256 `49116dee666c137f25bc76db5196326c5e64db20ce3ae9cd755b3f3e79b689e2`. The six source/split quotas match, no scene group crosses splits, and sampled image paths exist. Representative SIDD and PolyU validation pairs have matching dimensions and high pixel correlation.
- Split targets: SIDD 2,800/350/350 and PolyU 1,200/150/150 for train/validation/test. `scripts/prepare_mixed_dataset.py` assigns scenes to splits first and samples crops within each split; it must keep scenes separate.
- The previous fine-tuned checkpoints and result files cannot be recovered from the teammate's Windows run. Its notebook 03 output, with a different manifest SHA-256 (`574b2f63...`), is archived in `HeraldFolder/archive/`; do not cite it as current evidence. A `.venv` was created with Python 3.14 and CPU-only PyTorch. Nine tests pass. The three published pretrained checkpoints load and run. The official **real-denoising** Restormer checkpoint replaced the previously configured Gaussian checkpoint.
- The final 500-pair test uses identical **64 px center crops** for all three models. Results are in `HeraldFolder/test/` with per-image CSV and SHA-256-tagged summaries. PolyU PSNR: Restormer 40.0681 dB, NAFNet 39.1063 dB, DnCNN 37.7678 dB. Restormer is the current app recommendation. A 200-step DnCNN fine-tuning trial lowered 50-crop validation PSNR from 35.53753 to 32.26428 dB, so the published baseline was retained. The trial checkpoint and record are in `HeraldFolder/`.
- `HeraldFolder/FINAL_SUBMISSION.ipynb` executed locally top-to-bottom (7 code cells, 0 errors). `HeraldFolder/iMAGE_report.pdf` and `HeraldFolder/iMAGE_presentation.pptx` were generated from the measured summaries. The report PDF rendered cleanly; the PPTX has 10 slides and all shapes are within bounds. The app's real-mode upload endpoint returned HTTP 200, all three models, and the matching PolyU-based recommendation.
- The published Restormer real-denoising and NAFNet SIDD checkpoints were trained on SIDD training images. This project's SIDD subset comes from SIDD's training collection, so SIDD scores have pretraining overlap. Report results by source and emphasize PolyU for independent cross-dataset evidence. DnCNN's published checkpoint was trained for synthetic Gaussian noise.
- Data preparation and documentation changes are uncommitted. Check `git status` before editing; do not delete the ignored dataset or overwrite `data/splits.csv`.

## Next actions

1. Add the group number, member names, and verified individual contributions to `HeraldFolder/CONTRIBUTIONS.md`, report, and slides. Retain evidence of the instructor's verbal method clarification and NAFNet approval.
2. Upload the full project and untracked `data/` directory to Colab-accessible storage, run `HeraldFolder/FINAL_SUBMISSION.ipynb` top-to-bottom there, and save that executed Colab copy. This local execution does not prove Colab compatibility.
3. Review the report and slides for instructor formatting requirements. Published checkpoint training times are unavailable, so they are marked as such; the DnCNN pilot time is recorded separately. The SIDD pretraining overlap and 64 px crop limit must remain visible.
4. Package only the current submission notebook, three `methods/*.py` modules (plus required shared app code), report, slides, contribution statement, and dataset links. Do not include the archived notebook as current results or commit large ignored data/checkpoints by accident.
