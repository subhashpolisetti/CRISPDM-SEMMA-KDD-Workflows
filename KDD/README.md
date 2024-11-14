
# Social Network Ads Analysis Using KDD Methodology

## Project Overview

This project analyzes a social network advertising dataset using the **Knowledge Discovery in Databases (KDD)** methodology, which structures the steps from data selection to knowledge extraction. The aim is to build a predictive model to understand and forecast purchasing behavior based on demographic data. By employing machine learning techniques, we uncover key insights that can support targeted advertising strategies, helping businesses make data-driven decisions in marketing.

## Data Description

The dataset consists of three key features:
- **Age**: The age of each user in years.
- **Estimated Salary**: An estimate of the annual income for each user, in USD.
- **Purchased**: A binary target variable indicating whether a user purchased a product after viewing an ad (1 = purchased, 0 = not purchased).

These features help in building a model to identify the relationship between age, income, and purchasing tendencies, which can enhance the efficiency of social media advertisements.

## Project Workflow

Following the KDD methodology, the project includes five key phases:

### 1. Sample Phase

In the sampling phase, we begin by loading and examining the dataset to understand its structure, distribution, and overall characteristics. We then set up our environment by importing the necessary libraries:

```python
# Import libraries and load data
import pandas as pd
df = pd.read_csv('Social_Network_Ads.csv')
print("Dataset loaded successfully.")
print("Dataset Shape:", df.shape)
print("First few rows of the dataset:")
print(df.head())
```

Here, we inspect the shape of the data to ensure it’s complete and ready for analysis. Viewing the first few rows gives an idea of the data composition and quality, preparing us for in-depth exploration in the next phase.

### 2. Explore Phase

In this phase, we perform **exploratory data analysis (EDA)** to investigate feature distributions and visualize the relationships between features and the target variable, `Purchased`.

#### Age vs. Purchase Behavior

This plot helps us understand how age might influence purchasing decisions:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Plot Age vs. Purchase
sns.histplot(data=df, x='Age', hue='Purchased', kde=True, bins=30)
plt.title("Age Distribution by Purchase Decision")
plt.show()

#### Salary vs. Purchase Behavior

By analyzing the distribution of estimated salary, we can determine if income impacts the likelihood of purchasing after seeing an ad:

```python
# Plot Salary vs. Purchase
sns.histplot(data=df, x='EstimatedSalary', hue='Purchased', kde=True, bins=30)
plt.title("Salary Distribution by Purchase Decision")
plt.show()
```

This exploration phase reveals trends, such as middle-aged users or those with higher salaries showing a greater likelihood to make a purchase, which helps us later in feature selection.

### 3. Modify Phase

Data preprocessing is essential to prepare the data for machine learning. Here, we use **feature scaling** to standardize `Age` and `EstimatedSalary`, ensuring that each feature contributes proportionately to the model’s performance.

```python
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X = df[['Age', 'EstimatedSalary']]
X_scaled = scaler.fit_transform(X)
y = df['Purchased']
```

Standardizing the features helps the model converge more efficiently by adjusting different scales to a common scale, which is crucial in distance-based algorithms and helps avoid bias.

### 4. Model Phase

We apply a **Random Forest Classifier** to predict whether a user will purchase a product based on demographic data. The Random Forest algorithm is suitable here due to its robustness and ability to capture non-linear relationships.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Split data and train the model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Model training complete.")
```

Using `train_test_split`, we divide the data to train and evaluate the model. The `RandomForestClassifier` is fit to the training data, which learns patterns between the features and the target.

### 5. Assess Phase

Finally, we evaluate the model’s performance using accuracy, precision, recall, and a confusion matrix to ensure reliable predictions. This phase confirms the model’s effectiveness in identifying users likely to purchase after viewing an ad.

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict and evaluate
y_pred = rf_model.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:
", classification_report(y_test, y_pred))

# Confusion matrix visualization
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

The confusion matrix visualizes the model's ability to distinguish between users who did and did not purchase, helping us fine-tune and interpret the model’s success rate.

## Key Insights

The project provides the following insights:
- **Model Performance**: The Random Forest Classifier achieved an accuracy of **88.75%**, reflecting its effectiveness in predicting purchase behavior.
- **Feature Importance**: Analysis shows that both `Age` and `EstimatedSalary` play significant roles in determining a user's likelihood to make a purchase.
- **Behavioral Patterns**: Middle-aged users with higher salaries demonstrated higher purchase rates, suggesting demographic data can help enhance targeted advertising.

## How to Run the Project

To reproduce this project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/subhashpolisetti/CRISPDM-SEMMA-KDD-Workflows.git
   cd CRISPDM-SEMMA-KDD-Workflows
   ```


2. **Run the Notebook**:
   Open `Social_Network_Ads_Analysis_using_KDD_Process.ipynb` and execute each cell in order. The notebook includes code, visualizations, and explanations of each phase.

## Future Enhancements

Possible future directions for the project include:
- **Incorporating Additional Features**: Adding other demographic and behavioral features could further refine the model.
- **Implementing Other Models**: Experimenting with algorithms like Logistic Regression or XGBoost could provide more insights.
- **Real-Time Prediction**: Developing a real-time model with streaming data from social media platforms could enable dynamic ad targeting.



