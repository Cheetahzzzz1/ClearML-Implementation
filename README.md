# ClearML-Implementation

# Overview

This project implements a Convolutional Neural Network (CNN) to classify images from the FashionMNIST dataset using Pytorch. The training process is tracked using ClearML, which enables experiment logging, hyperparameter tracking and visualization of metrics.

# Setup and Installation

1. <ins> Clone the Repository </ins>

           git clone https://github.com/yourusername/FashionMNIST-ClearML.git
           cd FashionMNIST-CLearML

2. <ins> Install Dependencies </ins>

           pip install torch torchvision torchaudio
           pip install clearml matplotlib tqdm
   
4. <ins> Initialize ClearML </ins>

Before running the training script, configure ClearML:

            clearml-init

Follow the on-screen instructions to enter your ClearML API credentials.

# Training the Model

Run the training script to start model training and log metrics to ClearML:

             python scripts/train.py

This will :-

1. Load the FashionMNIST dataset.

2. Train CNN for 100 epochs.

3. Log accuracy, loss and system utilization in ClearML.

# Evaluating the Model

To evaluate the trained model on the test dataset, run:

             python scripts/evaluate.py

# Viewing ClearML Dashboard 

Once training is completed, go to ClearML Dashboard and navigate to:

1. Projects -> FashionMNIST -> Experiments

2. View Scalars, Logs, System Utilization and Configuration settings.

# Results

1. Final Training Accuracy : 98.88%

2. Final Test Accuracy : 92.33%

3. Logged Metrics : Loss, Accuracy, CPU/GPU usage
