# ChiDeK

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://iclr.cc/)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)]()

Data and code for the paper **"Learning Molecular Chirality via Chiral Determinant Kernels"** (ICLR 2026).

## Overview

**ChiDeK** (Chiral Determinant Kernels) is a framework that systematically integrates stereogenic information into molecular representation learning. Chirality is a fundamental molecular property that governs stereospecific behavior in chemistry and biology. Capturing chirality in machine learning models remains challenging due to the geometric complexity of stereochemical relationships and the limitations of traditional molecular representations that often lack explicit stereochemical encoding.

Existing approaches to chiral molecular representation primarily focus on central chirality, relying on handcrafted stereochemical tags or limited 3D encodings, and thus fail to generalize to more complex forms, such as axial chirality.

ChiDeK introduces:
- **Chiral determinant kernel**: Encodes the SE(3)-invariant chirality matrix
- **Cross-attention**: Integrates stereochemical information from local chiral centers into the global molecular representation
- **Unified architecture**: Jointly encodes central and axial chirality


## Requirements

- Python 3.10+
- PyTorch >= 2.0
- RDKit >= 2024.3
- [chiralfinder](https://github.com/Meteor-han/chiralfinder) (for chiral center/axis detection)
- pandas, numpy, scikit-learn, openpyxl


## Datasets

### Axial Chirality (included in `data/`)
- `hct_ecd_axial.pkl`: Axial chiral molecules with ECD spectra
- `hct_ecd_axial_res.pkl`: ChiralFinder results (run ChiralFinder.get_axial() to generate)
- `optical_rotation_589nm.csv`: Optical rotation at 589nm
- `ecd_axial_index_split.pkl`: Train/val/test split indices
- `axial_650.xlsx`: Chiral axis labels

### Central Chirality (to be prepared)
- **R/S classification**: Requires `RS_train.pkl`, `RS_validation.pkl`, `RS_test.pkl` with columns `rdkit_mol_cistrans_stereo`, `RS_label_binary`
- **Enantiomer ranking**: Requires `ranking_train.pkl`, `ranking_validation.pkl`, `ranking_test.pkl` with columns `rdkit_mol_cistrans_stereo`, `ID`, `top_score`
- **Central ECD**: Requires `hct_ecd.pkl` with peak number, position, height labels

Use `--data_dir` to specify the data directory (default: `data/`).

## Experiments

Run from the `src/` directory, examples are:

```bash
cd src
```

### 1. Central Chirality

**R/S classification**
```bash
python main_RS.py --bs 256 --lr 1e-4 --epochs 5 --use_qr --proj_dim 128 --hidden_dim 256 --num_layers 8 --device cuda:0
```

**Enantiomer ranking**
```bash
python main_ranking.py --bs 128 --lr 1e-4 --epochs 30 --use_qr --proj_dim 256 --hidden_dim 512 --num_layers 8 --device cuda:0
```

**ECD spectrum prediction (central)**
```bash
python main_ecd.py --bs 32 --lr 5e-4 --epochs 10 --use_qr --proj_dim 128 --hidden_dim 256 --num_layers 8 --device cuda:0
```

### 2. Axial Chirality

**ECD spectrum prediction (axial)**
```bash
python main_ecd_axial.py --bs 32 --lr 1e-4 --epochs 20 --use_qr --proj_dim 256 --hidden_dim 512 --num_layers 8 --device cuda:0
```

**Optical rotation prediction (axial)**
```bash
python main_optical_axial.py --bs 32 --lr 1e-4 --epochs 20 --use_qr --proj_dim 128 --hidden_dim 256 --num_layers 8 --device cuda:0
```


## Citation

```bibtex
@inproceedings{
shi2026learning,
  title={Learning Molecular Chirality via Chiral Determinant Kernels},
  author={Shi, Runhan and Zhang, Chi and Chen, Letian and Yu, Gufeng and Yang, Yang},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```
