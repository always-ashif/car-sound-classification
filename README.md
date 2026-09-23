# Motor Sound Classification Using CNN

## Overview

This project classifies motor sounds into **Diesel** and **Petrol** using **Audio Signal Processing, Librosa, and CNN**.

The audio files are processed using Librosa to extract numerical audio features and generate **Mel Spectrograms**. The Mel Spectrogram images are then used to train a CNN model.

> **Note:** This project uses a relatively small dataset, so the model's performance may not generalize well to larger and more diverse datasets.

## Dataset

The dataset was obtained from Kaggle.

**Kaggle Dataset:**
[Motor Sound Detect Dataset on Kaggle](https://www.kaggle.com/datasets/alwaysashif/motor-sound-detect?utm_source=chatgpt.com)

### Dataset Contents

| Class              | Samples |
| ------------------ | ------: |
| Diesel Motor Sound |     183 |
| Petrol Motor Sound |     137 |
| **Total**          | **320** |

## Audio Feature Extraction

Audio features were extracted using **Librosa**:

* Zero Crossing Rate (ZCR)
* RMS Energy
* MFCC
* Spectral Centroid
* Spectral Bandwidth
* Spectral Rolloff
* Chroma Features
* Mel Spectrogram

The numerical features were extracted for analysis, while Mel Spectrograms were converted into **2D images** for CNN training.

## Workflow

```text
Audio Files
     ↓
Librosa Feature Extraction
     ↓
Numerical Features + Mel Spectrogram
     ↓
Mel Spectrogram Images
     ↓
CNN
     ↓
Diesel / Petrol
```

## Data Preprocessing

Mel Spectrogram images were:

* Resized to **256 × 256**
* Rescaled from **0–255 to 0–1**

Dataset split:

* **80% Training**
* **10% Validation**
* **10% Testing**

## Model

A **Convolutional Neural Network (CNN)** was trained using the Mel Spectrogram images.

* Optimizer: **Adam**
* Loss: **Binary Crossentropy**
* Output activation: **Sigmoid**
* Metric: **Accuracy**

## Results

| Dataset    |   Accuracy |
| ---------- | ---------: |
| Training   | **84.38%** |
| Validation | **78.12%** |
| Testing    | **84.38%** |

The model was evaluated using **Accuracy, Classification Report, and Confusion Matrix**.

## Technologies

* Python
* TensorFlow / Keras
* Librosa
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Kaggle

## Author

**Ashif Ali**

GitHub: [@always-ashif](https://github.com/always-ashif)
