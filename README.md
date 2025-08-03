# Multi-Class-Image-Classification
This repository contains code for classifying fish images into multiple categories using a custom Convolutional Neural Network (CNN) and several pretrained models like VGG16, ResNet50, MobileNet, InceptionV3, and EfficientNetB0. The project includes preprocessing, training, evaluation, deployment, and performance comparison.

# Table of content 
---
- project Overview
- Installation
- Import
- Dataset
- Preprocessing
- Model
- Trainig
- Evaluation
- Result
- Usage
- Class Name
- Visualization

# project Overview

The objective of this project is to classify fish species using deep learning models. It compares the performance of a basic CNN and several pretrained models to identify which architecture performs best for fish image classification.

- **Dataset:** Images labeled into 11 fish categories.
- **Techniques:** Custom CNN, data augmentation, pretrained model fine-tuning (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0).
- **Evaluation:**  Metrics include accuracy, precision, recall, F1-score, and confusion matrix.
# Installation
```py
    pip install torch torchvision matplotlib numpy pandas seaborn scikit-learn tqdm
    pip install streamlit pyngrok streamlit-folium nbconvert
    npm install localtunnel
 ``` 