# HeartDiseasePredictorModel

This repository contains Python code to predict heart disease based on various attributes using a neural network model built with Keras and TensorFlow. Below is an overview of the code structure and the dataset used.

## Code Overview
### Files
* heart.csv: Dataset containing attributes related to heart health.
* heart_disease_prediction.py: Python script for data preprocessing, model training, evaluation, and visualization.

### Dependencies

* numpy: Fundamental package for numerical computing in Python.
* matplotlib: Library for creating static, animated, and interactive visualizations in Python.
* scikit-learn: Simple and efficient tools for predictive data analysis.
* keras: High-level neural networks API, running on top of TensorFlow.

### Usage
* Data Preparation: The dataset heart.csv is read and preprocessed to separate features (x) and target (y).
* Data Cleaning: Converts categorical variables into numerical values and performs data type conversion.
* Model Training: Builds a neural network model using Keras with specified layers and parameters.
* Model Evaluation: Trains the model on training data and evaluates its performance on test data using Mean Squared Error (MSE), Mean Absolute Error (MAE), and Accuracy metrics.
* Visualization: Displays training and validation loss, as well as training and validation accuracy over epochs.

### Dataset Columns
* Age: Age of the patient.
* Sex: Gender of the patient (0 = female, 1 = male).
* ChestPainType: Type of chest pain (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic).
* RestingBP: Resting blood pressure (mm Hg).
* Cholesterol: Serum cholesterol in mg/dl.
* FastingBS: Fasting blood sugar measured in mg/dl (1 = fasting blood sugar > 120 mg/dl; 0 = fasting blood sugar ≤ 120 mg/dl).
* RestingECG: Resting electrocardiographic results (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy).
    * 0 = Normal: normal ECG reading.
    * 1 = ST-T wave abnormality: signs of possible ischemia (lack of blood supply to the heart).
    * 2 = Left ventricular hypertrophy: enlargement and thickening of the heart's main pumping chamber.
* MaxHR: Maximum heart rate achieved.
* ExerciseAngina: Exercise-induced angina (chest pain or discomfort) during physical activity (1 = yes; 0 = no).
* Oldpeak: ST depression induced by exercise relative to rest, indicating the likelihood of coronary artery disease.
* ST_Slope: Slope of the peak exercise ST segment, categorized into three types
    * 1 = Upsloping: better prognosis for heart health.
    * 2 = Flat: minimal change (typical for healthy hearts).
    * 3 = Downsloping: signs of severe heart disease.
* HeartDisease: Presence of heart disease diagnosed based on angiography (0 = no heart disease, 1 = heart disease present).
