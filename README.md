# DEEP-LEARNING-PROJECT

COMPANY : CODTECH IT SOLUTIONS

NAME : NETRA ALLE

INTERN ID : CT04DG3270

DOMAIN : DATA SCIENCE

DURATION : 4 WEEKS

MENTOR : NEELA SANTOSH

MNIST Digit Classification Using PyTorch
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the popular MNIST dataset. It is a complete deep learning pipeline that includes dataset loading, preprocessing, model training, evaluation, result visualization, and model saving.

About the Dataset
The MNIST dataset is a benchmark dataset in machine learning, containing 60,000 training and 10,000 test images of handwritten digits (0–9). Each image is a 28x28 grayscale image, making it ideal for beginners to explore computer vision and deep learning.

Features
CNN architecture for digit recognition
Data augmentation using rotation and translation
Performance visualization (loss & accuracy plots)
Uses PyTorch with GPU/CPU compatibility
Confusion matrix to analyze classification errors
Sample prediction visualization with true/false coloring
Saves trained model using torch.save()

Why CNN for Image Classification?
Convolutional Neural Networks are highly effective for image-related tasks because they:
Use convolution layers to detect patterns like edges, corners, etc.
Reduce parameters compared to fully connected networks.
Handle spatial hierarchies using pooling layers.
Are scalable for high-dimensional data like images.

Requirements
To run this project, install the following Python packages:
bash
pip install torch torchvision matplotlib seaborn numpy scikit-learn
For Windows users (if you face issues with torch):

bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

How to Run the Code
Save the script as mnist_cnn.py.
Open terminal in the project directory.
Run the script:
bash
python mnist_cnn.py

Evaluation & Visualization
The model is trained over 10 epochs. After each epoch, it prints:
Training Loss
Training Accuracy
Test Loss
Test Accuracy

After training:
A line plot shows loss and accuracy over epochs.
A confusion matrix helps visualize model errors.
Five predicted digits are shown with their actual values, and green/red labels indicate correct or wrong predictions.

Output
mnist_cnn.pth: Saved PyTorch model.
Plots shown using matplotlib.
Prints training/testing logs in terminal.

Reusing the Saved Model
Later, you can reload the trained model with:
python
model.load_state_dict(torch.load('mnist_cnn.pth'))
model.eval()
This allows you to use the model for inference or deployment without retraining.

Model Architecture Summary
Input: (1, 28, 28)
→ Conv2D(1→32) → BatchNorm → ReLU → MaxPool
→ Conv2D(32→64) → BatchNorm → ReLU → MaxPool
→ Dropout
→ Flatten
→ FC(3136→128) → ReLU → Dropout
→ FC(128→10) → Output
File Structure
bash
├── mnist_cnn.py         # PyTorch training and evaluation script
├── mnist_cnn.pth        # Saved model
├── README.md            # Documentation

<img width="1493" height="701" alt="Image" src="https://github.com/user-attachments/assets/f09697e9-cea8-4abb-b0f2-48b798ef0f6e" />
<img width="1920" height="1020" alt="Image" src="https://github.com/user-attachments/assets/152b7212-4c9e-4fb4-9441-57d4a5ba0498" />
<img width="1765" height="447" alt="Image" src="https://github.com/user-attachments/assets/3faf2f75-cb91-4238-8e4e-292aa7fb16dc" />

