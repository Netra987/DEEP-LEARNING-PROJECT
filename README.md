# DEEP-LEARNING-PROJECT

COMPANY : CODTECH IT SOLUTIONS

NAME : NETRA ALLE

INTERN ID : CT04DG3270

DOMAIN : DATA SCIENCE

DURATION : 4 WEEKS

MENTOR : NEELA SANTOSH

MNIST Digit Classification Using PyTorch
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. It includes data preprocessing, model training, evaluation, and visualizations like accuracy plots and confusion matrix.

Features
Loads MNIST dataset with data augmentation (random rotation & translation)
CNN with batch normalization, dropout, and 2 convolution layers
Optimizer: Adam with learning rate scheduler (ReduceLROnPlateau)
Tracks training & testing accuracy and loss

Visualizes:
Loss and Accuracy curves
Confusion Matrix
Example predictions
Saves the trained model to mnist_cnn.pth

Requirements
Make sure you have Python 3.8+ installed.
You can install all necessary packages with:
pip install torch torchvision matplotlib seaborn numpy scikit-learn
If you're using Windows and face issues with torch, try this:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

How to Run
Clone or copy this repo into a folder.
Save the script as mnist_cnn.py
Open a terminal in that folder and run:
python mnist_cnn.py
You’ll see training logs, loss/accuracy plots, and prediction results.

Visual Outputs
Line charts showing loss and accuracy over 10 epochs
Confusion matrix heatmap of test predictions
5 prediction images with true/false color coding

Model Architecture
Input: (1, 28, 28)
↓ Conv2d(1, 32) → BatchNorm → ReLU → MaxPool
↓ Conv2d(32, 64) → BatchNorm → ReLU → MaxPool
↓ Dropout(0.25)
↓ Flatten → Linear(64*7*7 → 128) → ReLU → Dropout(0.25)
↓ Linear(128 → 10)
Output: Class probabilities for digits 0–9
📁 Project Structure
├── mnist_cnn.py           # Main PyTorch script
├── mnist_cnn.pth          # Saved model after training
├── README.md              # This file

Notes
Uses CPU or GPU automatically (cuda if available).
Handles data augmentation on training set only.
Adds batch normalization and dropout to improve generalization.
Saves the model for reuse/deployment.


