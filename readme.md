# Handwash Protocol Image Classifier

A deep learning image classification pipeline built in Python using TensorFlow and Keras to classify images into the 8 stages of the WHO hand hygiene protocol. The project applies transfer learning with EfficientNetB0, out-of-sample data cleaning using Cleanlab, and optimized data loading using `tf.data`.

## Overview

- **Input Resolution:** 150x150 RGB images
- **Classes:** 8 stages (Stage 1 to Stage 8)
- **Base Architecture:** Pretrained EfficientNetB0 (ImageNet weights)
- **Data Cleaning:** Cleanlab with 5-fold cross-validation to remove mislabeled images
- **Training Strategy:** Two-phase progressive training (frozen feature extractor followed by fine-tuning)

## Dataset & Preprocessing

1. **Dataset Normalization:** Images are gathered from class directories, converted to RGB, resized to 150x150, and assigned 0-indexed integer labels.
2. **Label Quality Control:** Feature vectors are extracted from the frozen backbone. Out-of-sample predicted probabilities are calculated using 5-fold stratified cross-validation. Cleanlab identifies and prunes label errors, reducing noise and improving data quality across the 8 stages.
3. **Data Splitting:** Stratified split into:
   - Train: 70%
   - Validation: 15%
   - Test: 15%
4. **Data Pipeline:** Implemented using `tf.data` with mapping, batching (batch size 32), caching, and asynchronous prefetching.

## Model Architecture & Training

- **Data Augmentation:** Random horizontal flips, slight rotation (0.05), and zoom (0.10).
- **Classification Head:** GlobalAveragePooling2D followed by 30% Dropout and an 8-unit Dense softmax output layer.
- **Phase 1 (Frozen Backbone):**
  - EfficientNetB0 backbone weights are frozen.
  - Optimizer: Adam (initial learning rate = 1e-3).
  - Callbacks: EarlyStopping, ModelCheckpoint (saving the best validation loss), and ReduceLROnPlateau.
- **Phase 2 (Fine-Tuning):**
  - Unfreezes the top layers of the backbone.
  - Optimizer: Adam with a lower learning rate (1e-5) to adjust weights without destroying pretrained feature representations.

## Evaluation

The model evaluates test set performance using:
- Accuracy and Categorical Cross-Entropy Loss
- Per-class Precision, Recall, and F1-score
- Confusion matrix and largest-confusion sample analysis

## Tech Stack

- Python
- TensorFlow / Keras
- Cleanlab
- scikit-learn
- NumPy
- Pandas
- Matplotlib
- Pillow
