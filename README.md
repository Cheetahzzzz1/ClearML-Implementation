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
