# Fake AI-Generated Face Detection

A binary image classifier built with TensorFlow/Keras that distinguishes real human face photos from AI-generated (fake) ones, using transfer learning with ResNet50.

## Overview

This project fine-tunes a pretrained **ResNet50** model to classify face images as **real** or **fake (AI-generated)**. It uses the [140k Real and Fake Faces dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) from Kaggle, which contains real-world resolution images with a built-in train/validation/test split.

## Dataset

- **Source:** `xhlulu/140k-real-and-fake-faces` (Kaggle)
- **Size:** ~140,000 images
- **Classes:** `real` vs. `fake`
- **Split:** Comes pre-split into train, validation, and test sets
- **Image size used:** 128x128 (resized from original resolution)

## Model Architecture

- **Base model:** ResNet50, pretrained on ImageNet, with the top classification layer removed
- **Approach:** Transfer learning in two stages:
  1. **Feature extraction** — base model frozen, only the new classification head is trained
  2. **Fine-tuning** — the last ~100 layers of ResNet50 are unfrozen and trained with a much lower learning rate (`1e-5`)
- **Data augmentation:** Random horizontal flip, rotation, and zoom applied to training images
- **Loss function:** Binary cross-entropy
- **Optimizer:** Adam

## Workflow

1. **Setup** – Configure Kaggle API credentials and download the dataset.
2. **Data loading** – Load images from directory into TensorFlow datasets, with prefetching for performance.
3. **Augmentation** – Apply random flips, rotations, and zooms to reduce overfitting.
4. **Feature extraction** – Train a classification head on top of the frozen ResNet50 base for 6 epochs.
5. **Evaluation** – Measure test accuracy and loss after feature extraction.
6. **Fine-tuning** – Unfreeze the top layers of ResNet50 and continue training at a lower learning rate for 6 more epochs.
7. **Final evaluation** – Re-evaluate on the test set after fine-tuning.
8. **Confusion matrix** – Visualize prediction performance across both classes.
9. **Single-image inference** – Run a prediction on one test image and display the predicted label with confidence.

## Requirements

```
tensorflow
kaggle
numpy
scikit-learn
matplotlib
```

Install with:

```bash
pip install tensorflow kaggle numpy scikit-learn matplotlib
```

## Setup

This notebook requires a Kaggle API key to download the dataset.

1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings) → **API** → **Create New Token** to download `kaggle.json`
2. **Do not hardcode your API key in the notebook.** Instead, store it securely:
   - In Colab: use `google.colab.userdata` to store `KAGGLE_USERNAME` and `KAGGLE_KEY` as secrets
   - Locally: place `kaggle.json` in `~/.kaggle/` (this file should never be committed to version control)

## Usage

Open `Fake_AI_generated_image_detection_new.ipynb` in Jupyter or Google Colab and run the cells in order. The notebook will:

1. Download and extract the dataset
2. Build TensorFlow data pipelines with augmentation
3. Train a classification head on frozen ResNet50 features
4. Fine-tune the top layers of ResNet50
5. Evaluate performance and display a confusion matrix
6. Run inference on a sample image

## Results

Test accuracy is reported both after initial feature-extraction training and after fine-tuning. See the notebook's output cells for exact values, along with the confusion matrix visualizing performance across the real/fake classes.

## License

This project is for educational purposes.
