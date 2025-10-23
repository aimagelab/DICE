# DICE: Detecting and Evaluating Instruction-Guided Image Edits

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://aimagelab.github.io/DICE)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2505.20405)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/aimagelab/DICE)

Official implementation of "What Changed? Detecting and Evaluating Instruction-Guided Image Edits with Multimodal Large Language Models"

## Overview

DICE (DIfference Coherence Estimator) is a novel framework designed to detect and evaluate instruction-guided image edits. It identifies differences between original and edited images and assesses their coherence with the editing prompt using Multimodal Large Language Models (MLLMs).

The framework consists of two main components:
1. **Difference Detector**: Identifies localized differences between the original and edited images
2. **Coherence Estimator**: Assesses the relevance of detected changes with respect to the editing prompt

## Key Features

- Object-level difference detection between image pairs
- Semantic coherence evaluation of edits
- Structured text generation for edit analysis
- High correlation with human judgment


## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{baraldi2025changed,
  title={What Changed? Detecting and Evaluating Instruction-Guided Image Edits with Multimodal Large Language Models},
  author={Baraldi, Lorenzo and Bucciarelli, Davide and Betti, Federico and Cornia, Marcella and Sebe, Nicu and Cucchiara, Rita and others},
  booktitle={Proceedings of the 2025 IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```

## Authors

- Lorenzo Baraldi - University of Pisa, Italy
- Davide Bucciarelli - University of Modena and Reggio Emilia, Italy
- Federico Betti - University of Trento, Italy
- Marcella Cornia - University of Modena and Reggio Emilia, Italy
- Lorenzo Baraldi - University of Modena and Reggio Emilia, Italy
- Nicu Sebe - University of Trento, Italy
- Rita Cucchiara - University of Modena and Reggio Emilia, Italy


## Code
# Simple Evaluation Bundle

This folder contains a self-contained copy of the assets required to run `simple_evaluation_example.py`



## Contents

- `simple_evaluation_example.py`: evaluation entry point.
- `original.jpg`, `edited.jpg`: sample original/edited images
  - `editing_evaluation/`: evaluation package.
  - `dataset/`: dataset utilities (includes coherence dataset loader).
  - `requirements.txt`, `README.md`: reference documentation and dependencies.

## Model Weights

Download the required model weights from:
https://huggingface.co/collections/aimagelab/dice

## Usage

Run the script from within this directory to keep relative paths valid:

```bash
cd simple_evaluation_bundle
python simple_evaluation_example.py
```


## Contact

For questions or issues, please open an issue in the GitHub repository.
