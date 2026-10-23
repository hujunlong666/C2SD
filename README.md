# C2SD

Graph classification training code for OGB molecular datasets and preprocessed DrugOOD datasets.

## Environment

The recorded Python environment uses Python 3.8.20, PyTorch 2.1.2+cu121, PyTorch Geometric 2.4.0 and OGB 1.3.6. See [C2SD_environment_versions.md](C2SD_environment_versions.md) for package versions.

## Data

Large datasets and model checkpoints are not included in this repository.

- OGB datasets: `ogbg-molbbbp`, `ogbg-molhiv`, and `ogbg-molbace`. The OGB loader uses the local `dataset/` directory and may download missing datasets.
- DrugOOD: place the preprocessed IC50/EC50 `.pt` files in `dataset/DrugOOD/`. Each `assay`, `scaffold`, or `size` setting requires a training file, an OOD validation file and an OOD test file.

Example DrugOOD filenames:

```text
drugood_lbap_core_ic50_scaffold_train.pt
drugood_lbap_core_ic50_scaffold_ood_val.pt
drugood_lbap_core_ic50_scaffold_ood_test.pt
```

## Run

Run commands from the repository root using an environment containing the recorded dependencies.

```bash
# OGB datasets
python main.py --dataset ogbg-molbbbp --device cuda --runs 3 --epochs 100
python main.py --dataset ogbg-molhiv --device cuda --runs 3 --epochs 100
python main.py --dataset ogbg-molbace --device cuda --runs 3 --epochs 100

# DrugOOD: IC50 and EC50 scaffold splits
python main.py --dataset drugood --drugood_prefix drugood_lbap_core_ic50 --drugood_ood scaffold --device cuda --runs 3 --epochs 100
python main.py --dataset drugood --drugood_prefix drugood_lbap_core_ec50 --drugood_ood scaffold --device cuda --runs 3 --epochs 100
```

For DrugOOD, replace `scaffold` with `assay` or `size` to select another split.

The training script saves the best validation-AUC checkpoint for each run under `checkpoints/` and writes a final `summary.json`. Early stopping defaults to 30 epochs without validation-AUC improvement; use `--early_stop_patience 0` to disable it.
