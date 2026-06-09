# COVID Hand Wash Stage Classification AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-lightgrey)
![Status](https://img.shields.io/badge/Academic_Grade-95%25-success)

An end-to-end Machine Learning pipeline designed to classify the distinct stages of the World Health Organization (WHO) handwashing protocol. This project implements a rigorous data-centric workflow, utilizing out-of-sample feature space filtering via Cleanlab to identify and remove mislabelled or noisy dataset samples prior to training a deep Convolutional Neural Network (CNN).

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Pipeline and Architecture Workflow](#pipeline-and-architecture-workflow)
5. [Dataset Cleaning Results](#dataset-cleaning-results)
6. [Installation and Setup](#installation-and-setup)
7. [Usage Instructions](#usage-instructions)
8. [Model Evaluation](#model-evaluation)

---

## Project Overview
* **Evaluation Score:** 95%
* **Objective:** Accurately categorize raw hand wash image frames into 8 sequential protocol stages.
* **Architecture:** Transfer Learning utilizing a pretrained EfficientNetB0 backbone combined with a customized Dense classification head.
* **Core Methodology:** Programmatic label noise identification via 5-fold cross-validated out-of-sample probability evaluations to isolate conflicting dataset points.

---

## Key Features
* **Data-Centric AI:** Moves beyond standard hyperparameter tuning by programmatically correcting raw data.
* **Optimized Data Pipeline:** Utilizes `tf.data` API for asynchronous prefetching, mapping, and caching, eliminating CPU/GPU bottlenecks.
* **Progressive Fine-Tuning:** Implements a two-phase training strategy to preserve pretrained weights before slowly unfreezing deeper convolutional blocks.
* **Automated Callbacks:** Integrates `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau` for optimal convergence.

---

## Project Structure

```text
handwash-protocol-image-classifier/
│
├── dataset/                                     # Directory containing the raw and cleaned image data
├── CODEBASE_DOCS.md                             # Detailed architecture and codebase documentation
├── handwash-protocol-image-classifier.ipynb     # Main executable Jupyter Notebook containing the full pipeline
└── readme.md                                    # Project documentation and overview