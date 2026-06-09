# Codebase Architecture and Documentation

This document provides a comprehensive breakdown of the codebase logic, module flow, and architectural decisions for the COVID Hand Wash Stage Classification project. It is intended to serve as a developer guide for understanding, reproducing, or extending the code found in the primary Jupyter Notebook (`Coursework2_ImageClassification_2566677.ipynb`).

---

## 1. Environment and Global Configuration
The codebase begins by establishing a reproducible environment and defining global constants.
* **Random Seeds:** NumPy, TensorFlow, and Python's native `random` module are seeded (e.g., `SEED = 42`) to ensure that train/test splits, model weight initializations, and cross-validation folds are entirely reproducible.
* **Global Variables:** Key hyperparameters are defined globally, including `IMG_SIZE = (150, 150, 3)`, `BATCH_SIZE = 32`, and `NUM_CLASSES = 8`.

---

## 2. Data Ingestion and Label Mapping
Because the raw data is organized in standard directories (one folder per class), the script utilizes a custom parsing loop rather than relying solely on `ImageDataGenerator`. This allows for tighter integration with the downstream Cleanlab processing.

1. **Path Collection:** `glob` is used to walk through the source directories.
2. **Label Encoding:** Directory names (`Stage_1` to `Stage_8`) are strictly parsed and mathematically mapped to a 0-indexed integer format (`0` to `7`). This is required for `SparseCategoricalCrossentropy` or one-hot encoded `CategoricalCrossentropy` loss functions.
3. **Dataframe Construction:** The absolute file paths and their corresponding integer labels are merged into a Pandas DataFrame to facilitate easy indexing and stratified splitting later.

---

## 3. Label Error Mitigation (The Cleanlab Pipeline)
The most computationally complex and distinct part of this codebase is the programmatic filtering of mislabelled data. This operates in three distinct phases:

### Phase A: Feature Extraction
To evaluate the images mathematically, they must be converted to vector embeddings.
* The script instantiates an `EfficientNetB0` model with `include_top=False` and `weights='imagenet'`.
* A `GlobalAveragePooling2D` layer is attached to flatten the spatial dimensions.
* The entire dataset is passed through this frozen backbone, resulting in a feature matrix of shape `(8538, 1280)`.

### Phase B: Out-of-Sample Probability Generation
Cleanlab requires unbiased probability predictions for every image in the dataset to detect label anomalies.
* A lightweight Multi-Layer Perceptron (MLP) or simple Dense network is constructed.
* The feature matrix is split using `StratifiedKFold` (typically 5 folds).
* The MLP is trained on 4 folds and predicts the probabilities of the 5th fold. This is repeated until every image has an out-of-sample prediction distribution.

### Phase C: Filtering
* `cleanlab.filter.find_label_issues` is invoked using the original noisy labels and the newly generated cross-validated probabilities.
* The function returns a boolean mask. The codebase filters the Pandas DataFrame using this mask, effectively dropping all rows flagged as label errors.

---

## 4. Dataset Splitting and tf.data Optimization
Once the dataset is cleaned, it is prepared for training.

* **Stratified Splitting:** `scikit-learn`'s `train_test_split` is utilized twice to create a `70% Training`, `15% Validation`, and `15% Testing` split, stratifying by the target label to ensure class balance across all sets.
* **Input Pipeline Construction:** The codebase utilizes the `tf.data.Dataset` API for maximum throughput:
  1. `from_tensor_slices()` maps the file paths and labels.
  2. A custom mapping function reads the image from the disk, decodes the JPEG/PNG, resizes it using high-fidelity LANCZOS/Bilinear interpolation to `(150, 150)`, and applies standard scaling (e.g., `preprocess_input` specific to EfficientNet).
  3. `cache()` is applied to store the images in RAM/disk after the first epoch.
  4. `batch(BATCH_SIZE)` aggregates the tensors.
  5. `prefetch(tf.data.AUTOTUNE)` allows the CPU to prepare the next batch while the GPU is training on the current batch.

---

## 5. Model Architecture
The primary classifier is built using the Keras Functional API.

* **Data Augmentation Layer:** A `Sequential` block containing `RandomFlip`, `RandomRotation`, and `RandomZoom` is placed at the head of the model. This ensures augmentations are only applied during training and execute on the GPU.
* **Base Model:** Pretrained `EfficientNetB0`. Chosen for its superior parameter efficiency and high baseline accuracy on ImageNet compared to standard ResNet architectures.
* **Classification Head:** * `GlobalAveragePooling2D` to reduce dimensionality.
  * `Dropout` (typically 0.3 to 0.5) to enforce regularization.
  * `Dense` layer with `softmax` activation outputting 8 class probabilities.

---

## 6. Progressive Training Strategy
To prevent the random initialization of the Dense classification head from corrupting the highly optimized ImageNet weights in the EfficientNet backbone, a two-phase training protocol is implemented.

### Phase 1: Warm-up
* **State:** Base model is completely frozen (`trainable = False`).
* **Optimizer:** Adam with a standard learning rate (e.g., `1e-3`).
* **Goal:** Allow the Dense head weights to converge toward reasonable values based on the extracted features.

### Phase 2: Fine-Tuning
* **State:** The top X layers (or specific convolutional blocks) of the EfficientNet base are unfrozen (`trainable = True`). Batch Normalization layers remain frozen to prevent statistics degradation.
* **Optimizer:** Adam with a strictly reduced learning rate (e.g., `1e-5`).
* **Goal:** Marginally adjust the feature extraction filters to become task-specific to human hands and water flow, without catastrophic forgetting.

### Automated Callbacks
Throughout both phases, the codebase relies on:
* `ModelCheckpoint`: Monitors `val_loss` and saves only the optimal weight states.
* `EarlyStopping`: Halts training if `val_loss` fails to improve for a set number of epochs (patience).
* `ReduceLROnPlateau`: Divides the learning rate by a factor if validation metrics stagnate, allowing the optimizer to settle into narrow local minima.

---

## 7. Inference and Evaluation Modules
The final section of the codebase handles rigorous performance tracking.

* **Classification Report:** Generates Precision, Recall, and F1-scores for all 8 stages utilizing `sklearn.metrics`.
* **Confusion Matrix:** Plots a normalized heatmap. This is critical for visual inspection of boundary errors (e.g., the model confusing Stage 2 with Stage 3).
* **External Inference Function:** A standalone Python function defined to take a single, unseen `.jpg` file from a local directory, apply the exact `tf.data` preprocessing steps, run a forward pass through the trained model, and print the predicted class alongside the softmax confidence array.