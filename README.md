## Multi-Class-Image-Classification
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

 Imports

 ```py
 # File Handling and Image Processing
import os
import zipfile
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random

# Torch and Vision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Display and Progress
from IPython.display import display
from tqdm import tqdm

# Data Handling
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Warnings
import warnings
warnings.filterwarnings('ignore')
 ```
 # Dataset
 The dataset contains labeled fish images categorized into:

-  Train: Training data
- val: Validation data
- test: Testing data

# Preprocessing
- Resize images to 224x224
- Random horizontal flips
- Rotation between -15° to 15°
- Random affine transforms
- Normalize using ImageNet mean and std

# Model

# CNN model
- 2 convolutional layers + ReLU + MaxPooling
- Dropout layers
- Fully connected layers for classification

# Pretrained Models
- VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0
- Custom final layers adapted to 11 classes

`py
vgg16.classifier[6] = nn.Linear(num_features, 11)`

