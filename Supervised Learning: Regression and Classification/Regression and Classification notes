# Supervised Learning: Regression and Classification

Welcome to the repository! This repository contains Python implementations of supervised learning models, including:
- **Single Variable Linear Regression**
- **Single Variable Logistic Regression**
- **Multi-Variable Linear Regression**
- **Multi-Variable Logistic Regression**

All models are implemented from scratch to give a deeper understanding of how these algorithms work. Below, you will find the problem-solving pipeline, relevant formulas, and links to resources that will help you gain a strong foundation in machine learning.

---

## 📚 Table of Contents

- [Introduction](#introduction)
- [Formulas](#formulas)
- [Problem Solving Pipeline](#problem-solving-pipeline)
- [Model Implementations](#model-implementations)
  - [Single Variable Linear Regression](#single-variable-linear-regression)
  - [Single Variable Logistic Regression](#single-variable-logistic-regression)
  - [Multi-Variable Linear Regression](#multi-variable-linear-regression)
  - [Multi-Variable Logistic Regression](#multi-variable-logistic-regression)
- [Resources](#resources)

---

## Introduction

In supervised learning, we aim to map an input `X` to an output `Y` based on a dataset. The two main categories of supervised learning are:

1. **Regression**: Predicting continuous values (e.g., house prices, stock prices).
2. **Classification**: Predicting categorical labels (e.g., spam or not spam).

This repository focuses on implementing the core regression and classification algorithms using Python and NumPy.

### Key Concepts:
- **Gradient Descent**: An optimization algorithm to minimize the cost function by adjusting the weights.
- **Cost Functions**: Measures how well the model fits the data.
  - Mean Squared Error (MSE) for regression
  - Cross-Entropy (Log Loss) for classification

---

## 📐 Formulas

### Linear Regression (Single Variable)

1. **Hypothesis**:  
   \[
   f_{w,b}(x) = w \cdot X + b
   \]
   
2. **Cost Function (Mean Squared Error)**:  
   \[
   J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2
   \]
   
3. **Gradient Descent Update Rule**:
   - For weights \( w \):
     \[
     w := w - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) \cdot x^{(i)}
     \]
   - For bias \( b \):
     \[
     b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)
     \]

### Logistic Regression (Single Variable)

1. **Hypothesis (Sigmoid Function)**:  
   \[
   f_{w,b}(x) = \frac{1}{1 + e^{-(w \cdot X + b)}}
   \]

2. **Cost Function (Binary Cross-Entropy Loss)**:  
   \[
   J(w, b) = - \frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f_{w,b}(x^{(i)})) + (1 - y^{(i)}) \log(1 - f_{w,b}(x^{(i)})) \right]
   \]

3. **Gradient Descent Update Rule**:
   - For weights \( w \):
     \[
     w := w - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) \cdot x^{(i)}
     \]
   - For bias \( b \):
     \[
     b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)
     \]

### Linear Regression (Multi-Variable)

1. **Hypothesis**:  
   \[
   f_{w,b}(X) = W \cdot X + b
   \]

2. **Cost Function (Mean Squared Error)**:  
   \[
   J(W, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(X^{(i)}) - y^{(i)} \right)^2
   \]

3. **Gradient Descent Update Rule**:
   - For weights \( W \):
     \[
     W := W - \alpha \frac{1}{m} X^T \left( f_{w,b}(X) - Y \right)
     \]
   - For bias \( b \):
     \[
     b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(X^{(i)}) - y^{(i)} \right)
     \]

### Logistic Regression (Multi-Variable)

1. **Hypothesis (Sigmoid Function)**:  
   \[
   f_{w,b}(X) = \frac{1}{1 + e^{-(W \cdot X + b)}}
   \]

2. **Cost Function (Binary Cross-Entropy Loss)**:  
   \[
   J(W, b) = - \frac{1}{m} \sum_{i=1}^{m} \left[ Y^{(i)} \log(f_{w,b}(X^{(i)})) + (1 - Y^{(i)}) \log(1 - f_{w,b}(X^{(i)})) \right]
   \]

3. **Gradient Descent Update Rule**:
   - For weights \( W \):
     \[
     W := W - \alpha \frac{1}{m} X^T \left( f_{w,b}(X) - Y \right)
     \]
   - For bias \( b \):
     \[
     b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(X^{(i)}) - Y^{(i)} \right)
     \]

---

## 🛠️ Problem Solving Pipeline

For solving regression or classification problems, you can follow this general pipeline:

1. **Understand the Problem**:
   - Is it a **regression** problem (continuous output) or a **classification** problem (categorical output)?

2. **Collect and Explore the Data**:
   - Load your dataset.
   - Explore the data: Check for missing values, outliers, and correlations between variables.
   - Split the dataset into training and testing sets.

3. **Preprocess the Data**:
   - Standardize or normalize features (if necessary).
   - Handle missing data through techniques like **mean imputation**, **median imputation**, or using advanced methods.

4. **Select an Algorithm**:
   - Use **Linear Regression** if your target is continuous.
   - Use **Logistic Regression** if your target is binary or categorical.

5. **Train the Model**:
   - Initialize weights and bias.
   - Perform **gradient descent** to minimize the cost function.
   - Tune hyperparameters such as the learning rate `α` and the number of iterations.

6. **Evaluate the Model**:
   - For **regression**, use **Mean Squared Error (MSE)** or **Mean Absolute Error (MAE)**.
   - For **classification**, use **accuracy**, **precision**, **recall**, and **F1 score**.

7. **Make Predictions**:
   - Use the trained model to make predictions on new data.
   - For regression, predict a continuous value.
   - For classification, predict the class label.

8. **Test the Model**:
   - Evaluate the model performance on the testing dataset.
   - Use metrics appropriate to the problem type.

9. **Iterate**:
   - Tweak the model by experimenting with different features, learning rates, or number of iterations.
   - Re-train and re-evaluate as needed.

---

## 💻 Model Implementations

### Single Variable Linear Regression

- Predicts a continuous value using a single input variable.
- **Implementation File**: `single_variable_linear.py`
- **Formula**:
  \[
  f_{w,b}(x) = w \cdot X + b
  \]

### Single Variable Logistic Regression

- Predicts a binary output using a single input variable.
- **Implementation File**: `single_variable_logistic.py`
- **Formula**:
  \[
  f_{w,b}(x) = \frac{1}{1 + e^{-(w \cdot X + b)}}
  \]

### Multi-Variable Linear Regression

- Predicts a continuous value using multiple input variables.
- **Implementation File**: `multi_variable_linear.py`
- **Formula**:
  \[
  f_{w,b}(X) = W \cdot X + b
  \]

### Multi-Variable Logistic Regression

- Predicts a binary output using multiple input variables.
- **Implementation File**:

 `multi_variable_logistic.py`
- **Formula**:
  \[
  f_{w,b}(X) = \frac{1}{1 + e^{-(W \cdot X + b)}}
  \]

---

## 🔗 Resources

- [Andrew Ng's Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) - Learn more about the algorithms used in these models.
- [NumPy Documentation](https://numpy.org/doc/stable/) - For understanding the NumPy library used in these implementations.
- [Gradient Descent Visualizer](https://towardsdatascience.com/gradient-descent-algorithm-and-its-variants-10f652806a3) - A blog explaining gradient descent with visualizations.
