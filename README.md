# Fish Image Classification Project

This project is a fish species image classification task, using a dataset from Kaggle. The aim is to classify various species of fish using a neural network built with TensorFlow and Keras. The project involves training an artificial neural network (ANN) model and evaluating its performance on a test dataset.

## Project Summary

* **Dataset**: Kaggle Fish Species Dataset
* **Model**: Artificial Neural Network (ANN) built using TensorFlow and Keras
* **Task**: Multi-class classification of fish species
* **Classes**: 9 different fish species
* **Evaluation Metrics**: Accuracy, Confusion Matrix, Loss Curves, and Classification Report

## Model Architecture

The model is built using TensorFlow and Keras. The core of the model is an Artificial Neural Network (ANN) with several dense layers, using the ReLU activation function and softmax activation for the output layer.

* **Input Layer**: Flattened image input
* **Hidden Layers**: Dense layers with ReLU activation
* **Output Layer**: Dense layer with softmax for multi-class classification

The dataset was split into training and testing sets, and the training set was further split for validation purposes during model training.

## Performance Metrics
### Final Model Evaluation

* **Test Accuracy**: 94.2%
* **Test Loss**: 0.1602

### Confusion Matrix 

The confusion matrix for the test dataset highlights the performance of the model in predicting different species. The confusion matrix shows a high accuracy rate across most fish species, with minimal misclassifications.

### Loss and Accuracy Curves 

The model's training and validation performance over the epochs is:
* **Loss Curves**: Both training and validation losses decrease consistently over time, indicating effective learning.
* **Accuracy Curves**: Both training and validation accuracy exceeded 90%, with minimal overfitting.

## Techniques Used for Performance Improvement 

1. **Model Architecture Tuning**:

* Used techniques like Dropout to prevent overfitting and improve generalization.

2. **Optimization**:

* Used the Adam optimizer with a tuned learning rate.
* Early Stopping was employed to avoid overfitting, monitoring the validation loss.

3. **Normalization**:

* Pixel values were normalized to speed up convergence and ensure better generalization.

## Results and Insights

* The model achieved a high test accuracy of 94.2%, demonstrating its capability to classify fish species effectively.
* The use of optimization and normalization improved the model's performance and prevented overfitting.
* The confusion matrix shows that some species (e.g., Trout, Red Mullet) are more prone to misclassification, likely due to visual similarities between species.
* Further improvements could be explored by integrating convolutional layers (CNN) to potentially boost the model’s feature extraction capabilities.

## Running the Code
### Prerequisites

To run this project, you will need:

* Python 3.x
* TensorFlow 2.x
* Keras
* scikit-learn
* seaborn (for visualizations)
* matplotlib

## Kaggle
[Kaggle Website](https://www.kaggle.com/code/senakivrak/classification-project-deep-learning-ann/notebook)

