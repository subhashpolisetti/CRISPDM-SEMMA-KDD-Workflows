# Rainfall Prediction Using Machine Learning - A CRISP-DM Approach

## Table of Contents
1. [Introduction](#introduction)
2. [CRISP-DM Methodology](#crisp-dm-methodology)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)

## Introduction

This project applies the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology to predict rainfall using machine learning. Accurate rainfall predictions have practical applications in fields like agriculture, disaster preparedness, and urban planning. Using weather data, we analyze key features, build a predictive model, and evaluate its effectiveness in forecasting rainfall.

## CRISP-DM Methodology

We use the CRISP-DM methodology to structure the project, covering each of the six phases:

1. **Business Understanding**: Define project goals from a business perspective.
2. **Data Understanding**: Gather and explore data to identify patterns and insights.
3. **Data Preparation**: Clean and preprocess data to make it ready for modeling.
4. **Modeling**: Select and train a machine learning model to predict rainfall.
5. **Evaluation**: Assess the model's performance with relevant metrics.
6. **Deployment**: Discuss potential deployment strategies for practical applications.

## Dataset

The dataset used for this project includes weather observations with the following features:
- **Temperature (°C)**: Ambient temperature
- **Humidity (%)**: Atmospheric humidity
- **Wind Speed (km/h)**: Speed of the wind
- **Cloud Cover (%)**: Cloudiness in percentage
- **Pressure (hPa)**: Atmospheric pressure

You can download the dataset from [Kaggle](https://www.kaggle.com/zeeshier/weather-forecast-dataset).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/subhashpolisetti/CRISPDM-SEMMA-KDD-Workflows.git
   cd CRISPDM-SEMMA-KDD-Workflows
   ```



2. Download the Kaggle dataset:
   - Create a kaggle.json file with your Kaggle API credentials
   - Upload kaggle.json to the project directory
   - Run the following commands:
   ```python
   from google.colab import files
   files.upload()  # Upload kaggle.json
   !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   !kaggle datasets download -d zeeshier/weather-forecast-dataset -p ./
   !unzip ./weather-forecast-dataset.zip -d ./
   ```

## Usage

This section describes each CRISP-DM step and how to execute it.

### Step 1: Business Understanding
The project's objective is to create a model that can predict rainfall based on weather parameters. This model aims to assist in decision-making in sectors affected by rainfall, such as agriculture and urban planning.

### Step 2: Data Understanding
In the notebook `CRISP_DM_Weather_Analysis.ipynb`, we perform data exploration to gain insights into the data:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(data, hue='Rain', diag_kind='hist')
plt.show()
```

### Step 3: Data Preparation
Run the following script for data preprocessing:
```bash
python src/data_preprocessing.py
```

### Step 4: Modeling
Train the Random Forest model:
```bash
python src/model.py
```

Example code:
```python
from sklearn.ensemble import RandomForestClassifier

# Train model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_scaled, y_train)
```

### Step 5: Evaluation
```python
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

y_pred = rf_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

### Step 6: Deployment
Future considerations include:
- Deploying the model as an API for real-time predictions
- Model monitoring and retraining
- Integration with weather forecasting systems

## Results

The model achieved high accuracy in predicting rainfall, with Humidity and Cloud Cover identified as the most important predictors.



