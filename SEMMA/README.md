
# **Spotify Song Recommendation System Using SEMMA Methodology**

## Project Overview

This project applies the **SEMMA methodology** (Sample, Explore, Modify, Model, Assess) to develop a Spotify Song Recommendation System. Using Spotify’s audio features dataset, the system predicts whether a user will like a song based on various characteristics, such as danceability, energy, and valence.

## Data Description

The dataset includes a variety of audio features extracted from songs, such as:

- **Danceability**: Suitability for dancing (0.0 to 1.0)
- **Energy**: Intensity and activity measure (0.0 to 1.0)
- **Loudness**: Overall loudness in decibels (dB)
- **Speechiness**: Measure of spoken words (0.0 to 1.0)
- **Acousticness**: Confidence level of acoustic sound (0.0 to 1.0)
- **Instrumentalness**: Likelihood of a track containing no vocals (0.0 to 1.0)
- **Liveness**: Detects presence of an audience (0.0 to 1.0)
- **Valence**: Musical positiveness (0.0 to 1.0)
- **Tempo**: Track's tempo in BPM
- **Target Variable**: `liked` (binary classification, 1 for liked and 0 for not liked)

## Project Workflow

The project is organized according to the SEMMA methodology, covering each phase:

### 1. Sample Phase

The sampling phase focuses on loading the data and examining its structure. Here, we import the necessary libraries and load the dataset:

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
from sklearn.decomposition import PCA

# Load the dataset
print("Loading the dataset...")
df = pd.read_csv('data.csv')
print("Dataset loaded successfully.")
```

#### Check Dataset Structure

```python
# Display the shape and class distribution
print("\n1. SAMPLE PHASE")
print("Original dataset shape:", df.shape)
print("\nClass distribution:")
print(df['liked'].value_counts(normalize=True))
```

### 2. Explore Phase

During this phase, we analyze the dataset's distribution and identify potential relationships between features:

```python
# Visualize feature distributions
df.hist(bins=20, figsize=(20, 15))
plt.show()

# Correlation heatmap to check relationships
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.show()
```

### 3. Modify Phase

This phase focuses on data preprocessing, such as scaling features and reducing dimensionality using PCA (Principal Component Analysis):

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scale the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop(columns=['liked']))

# Apply PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Visualize PCA components
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['liked'], cmap='viridis')
plt.title("PCA of Audio Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()
```

### 4. Model Phase

The model phase involves training a Random Forest Classifier on the preprocessed data:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Prepare data for training
X = df.drop(columns=['liked'])
y = df['liked']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Model training complete.")
```

#### Feature Importance

We assess which features contribute most to the model's prediction:

```python
# Display feature importance
importances = model.feature_importances_
sns.barplot(x=importances, y=X.columns)
plt.title("Feature Importance in Predicting Song Likability")
plt.show()
```

### 5. Assess Phase

Evaluate model performance using metrics like accuracy, precision, recall, and confusion matrix:

```python
from sklearn.metrics import classification_report, confusion_matrix

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix of Song Likability Predictions")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
```

## How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/subhashpolisetti/CRISPDM-SEMMA-KDD-Workflows.git
   cd RISPDM-SEMMA-KDD-Workflows
   ```

2. **Upload Kaggle API Key**:
   - Download your `kaggle.json` API key from Kaggle.
   - Run the following code in your Colab notebook to upload it:

   ```python
   from google.colab import files
   files.upload()  # Choose kaggle.json when prompted

   # Set up Kaggle API for data download
   !mkdir -p ~/.kaggle
   !mv kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json

   # Download and extract dataset
   !kaggle datasets download -d bricevergnou/spotify-recommendation -p ./
   !unzip ./spotify-recommendation.zip -d ./
   ```

3. **Run Each Code Cell**:
   - Follow the notebook sections in Google Colab, running each code cell sequentially to complete the SEMMA methodology.

## Results

The model achieved an **accuracy of 89%**, with balanced precision and recall metrics. This demonstrates the model's potential in accurately recommending songs that align with user preferences.

## Future Work

Future improvements include:
- **Integrating Deep Learning**: Using neural networks to capture complex feature interactions.
- **Time-Series Analysis**: Including time-based preferences for more dynamic recommendations.
- **Cross-Domain Recommendations**: Expanding to include other media types, like podcasts.





