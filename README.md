# Digit Recognition Project

## Overview

This project is a digit recognition system that uses a neural network to classify handwritten digits from the MNIST dataset. The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0 through 9), and the goal is to accurately classify each image.

## Features

- **Handwritten Digit Classification**: Classifies images of handwritten digits from 0 to 9.
- **Data Preprocessing**: Normalizes image pixel values to the range [0, 1] for better neural network performance.
- **Model Architecture**:
  - **Initial Model**: A simple neural network with one dense layer.
  - **Improved Model**: A more complex neural network with additional hidden layers and ReLU activation function for improved accuracy.
- **Evaluation**: Measures model performance using accuracy and loss metrics. Confusion matrix visualizations provide insights into classification errors.

## Technologies

- **Python**: Programming language used for implementation.
- **TensorFlow & Keras**: Deep learning framework used to build and train neural network models.
- **NumPy**: Used for numerical operations and data manipulation.
- **Pandas**: Utilized for handling and preprocessing data.
- **Matplotlib**: For visualizing images and training results.
- **Seaborn**: For creating a heatmap of the confusion matrix.

## How It Works

1. **Data Loading**: Load the MNIST dataset using `keras.datasets.mnist.load_data()`.
2. **Data Preprocessing**:
   - Normalize the pixel values of images to the range [0, 1] by dividing by 255.
   - Flatten the images into 784-dimensional vectors (28x28 pixels).
3. **Model Building**:
   - **Initial Model**: A single dense layer with sigmoid activation function.
   - **Improved Model**: A sequential model with:
     - Flatten layer to reshape input data.
     - Dense hidden layer with ReLU activation function.
     - Output dense layer with 10 units (one for each digit) and sigmoid activation.
4. **Model Compilation**:
   - Use Adam optimizer.
   - Sparse categorical crossentropy loss function.
   - Track accuracy as a metric.
5. **Model Training**: Train the model using the training data for a specified number of epochs.
6. **Evaluation**: Test the model on unseen test data and evaluate performance.
7. **Prediction**: Make predictions on test data and visualize results.
8. **Confusion Matrix**: Generate and visualize the confusion matrix to analyze classification errors.


