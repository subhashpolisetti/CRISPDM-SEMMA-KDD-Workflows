
# Mining Methodologies: CRISP-DM, KDD, and SEMMA

This repository demonstrates three data mining methodologies applied to various real-world datasets: **CRISP-DM** (Weather Analysis), **KDD** (Social Media Ads Analysis), and **SEMMA** (Spotify Recommendation System). Each project includes data exploration, preprocessing, modeling, and evaluation steps, along with comprehensive documentation and supporting files.

---

## Project 1: CRISP-DM - Weather Analysis

### Overview
The Weather Analysis project aims to predict rainfall based on meteorological data using machine learning techniques. Accurate rainfall prediction can assist in planning weather-dependent activities. This project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology.

### Dataset
The dataset used is the **[Weather Forecast Dataset](https://www.kaggle.com/zeeshier/weather-forecast-dataset)**, which includes temperature, humidity, wind speed, cloud cover, pressure, and rainfall status (target variable).
- **Rows**: 10,000+ observations
- **Columns**: Temperature, humidity, wind speed, etc., plus rainfall as the target variable

### CRISP-DM Process Steps
1. **Business Understanding**: The goal is to predict whether it will rain, aiding planning in weather-sensitive fields.
2. **Data Understanding**: Explored features like temperature, humidity, and pressure to understand their impact on rainfall.
3. **Data Preparation**: Preprocessed data by handling missing values, encoding categorical variables, and scaling features.
4. **Modeling**: Trained several models, selecting a **Decision Tree Classifier** with an accuracy of **78%**.
5. **Evaluation**: Evaluated model accuracy, precision, recall, and F1-score.
6. **Deployment**: Saved the trained model using `joblib` for deployment.

### Additional Resources
- **Colab Notebook**: [CRISP_DM_Weather_Analysis Colab File Link ](https://github.com/subhashpolisetti/CRISPDM-SEMMA-KDD-Workflows/blob/main/CRISP%20DM/CRISP_DM_Weather_Analysis.ipynb)
- **Medium Article**: [Building a Credit Card Fraud Detection System using the SEMMA Process](https://medium.com/@subhashr161347/predicting-rainfall-using-machine-learning-a-crisp-dm-approach-2470865abfd3)
- **Research Paper**: [CRISP_DM_Weather_Analysis Research Paper Link](https://github.com/subhashpolisetti/CRISPDM-SEMMA-KDD-Workflows/blob/main/CRISP%20DM/CRISP_DM_Weather_Analysis.pdf)
- **LaTeX Format Code**: [CRISP LaTeX Format](https://github.com/subhashpolisetti/CRISPDM-SEMMA-KDD-Workflows/blob/main/CRISP%20DM/CRISP_DM_Weather_Analysis.tex)

---

## Project 2: KDD - Social Media Ads Analysis

### Overview
The Social Media Ads Analysis project examines advertising data to predict user purchasing behavior based on demographic features. Using the **KDD (Knowledge Discovery in Databases)** methodology, the project identifies key factors influencing purchase behavior, supporting targeted ad strategies.

### Dataset
The dataset used is the **[Social Network Ads Dataset](https://www.kaggle.com/datasets/srinivasatul/social-network-ads)**, containing age, estimated salary, and purchase status (target variable).
- **Rows**: 1,000+ users
- **Columns**: Age, estimated salary, and purchase status

### KDD Process Steps
1. **Selection**: Selected key demographic features (age, salary, purchase status).
2. **Preprocessing**: Handled missing values and standardized numerical features.
3. **Transformation**: Encoded necessary categorical variables and standardized data.
4. **Data Mining**: Trained multiple models, selecting a **Decision Tree Classifier** with the best performance.
5. **Evaluation**: Used confusion matrix, accuracy, and precision to evaluate predictions.

### Additional Resources
- **Colab Notebook**: [Social_Network_Ads_Analysis_using_KDD_Process Colab File Link](https://github.com/subhashpolisetti/CRISPDM-SEMMA-KDD-Workflows/blob/main/KDD/Social_Network_Ads_Analysis_using_KDD_Process.ipynb)
- **Medium Article**: [Building a Credit Card Fraud Detection System using the SEMMA Process](https://medium.com/@subhashr161347/using-the-kdd-process-for-social-network-ads-analysis-24c6770ef2bf)
- **Research Paper**: [Social_Network_Ads_Analysis_using_KDD_Process Research Paper](https://github.com/subhashpolisetti/CRISPDM-SEMMA-KDD-Workflows/blob/main/KDD/Social_Network_Ads_Analysis_using_KDD_Process.pdf)
- **LaTeX Format Code**: [Social_Network_Ads_Analysis_using_KDD_Process LaTeX Format](https://github.com/subhashpolisetti/CRISPDM-SEMMA-KDD-Workflows/blob/main/KDD/Social_Network_Ads_Analysis_using_KDD_Process.tex)

---

## Project 3: SEMMA - Spotify Recommendation System

### Overview
This project uses audio data from Spotify to predict song likability, enhancing music recommendation systems. Using the **SEMMA (Sample, Explore, Modify, Model, Assess)** methodology, the analysis identifies song characteristics influencing user preferences.

### Dataset
The **[Spotify Recommendation Dataset](https://www.kaggle.com/bricevergnou/spotify-recommendation)** includes audio features like danceability, energy, loudness, speechiness, and liked status (target variable).
- **Features**: Danceability, energy, valence, tempo, and liked status.

### SEMMA Process Steps
1. **Sample**: Sampled the dataset to capture the range of song characteristics.
2. **Explore**: Analyzed relationships among features such as danceability, energy, and valence.
3. **Modify**: Scaled and engineered features based on audio insights.
4. **Model**: Trained several models, selecting a **Random Forest Classifier** with **82% accuracy**.
5. **Assess**: Evaluated the model using precision, recall, and F1-score.

### Additional Resources
- **Colab Notebook**: [Spotify_Recommendation_System_using_SEMMA Colab File](https://github.com/subhashpolisetti/CRISPDM-SEMMA-KDD-Workflows/blob/main/SEMMA/Spotify_Recommendation_System_using_SEMMA.ipynb)
- **Medium Article**: [Spotify_Recommendation_System_using_SEMMA](https://medium.com/@subhashr161347/spotify-song-recommendation-system-using-semma-methodology-c436cb112f91)
- **Research Paper**: [Spotify_Recommendation_System_using_SEMMA Research Paper](https://github.com/subhashpolisetti/CRISPDM-SEMMA-KDD-Workflows/blob/main/SEMMA/Spotify_Recommendation_System_using_SEMMA.pdf)
- **LaTeX Format Code**: [Spotify_Recommendation_System_using_SEMMA LaTeX Format](https://github.com/subhashpolisetti/CRISPDM-SEMMA-KDD-Workflows/blob/main/SEMMA/Spotify_Recommendation_System_using_SEMMA.tex)

