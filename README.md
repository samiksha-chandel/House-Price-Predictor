# 🏡 Ames House Price Predictor

A web application built with **Streamlit** and powered by an **XGBoost Regressor** model to predict house prices in Ames, Iowa. Just enter the property details and get an estimated price along with confidence intervals, comparison plots, and interactive visualizations.

---

## 🚀 Features

- 💰 Predicts house prices based on 9 important features
- 📏 Shows 95% confidence interval for the prediction
- 📍 Interactive map showing the neighborhood
- 📊 Radar chart comparing user input with average house features
- 📁 Batch prediction from uploaded CSV files
- ✅ Clean UI with error handling and visual feedback

---

## 🧠 Model Overview

The app uses a pre-trained **XGBoost Regressor**, trained on the [Ames Housing dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset) with:
- **Log-transformed prices** (for stability)
- **9 curated features** that strongly impact price
- Scaled input using `StandardScaler`

---

## 📁 Files in This Repo

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit app that powers the UI and prediction |
| `train_model.py` | (Optional) Code to train and export the XGBoost model |
| `xgb_model.joblib` | Trained XGBoost model |
| `scaler.joblib` | Scaler used to normalize input data |
| `features.joblib` | List of selected features |
| `AmesHousing.csv` | (Optional) Original dataset used for training |
| `requirements.txt` | List of Python dependencies |
| `sample_input.csv` | Sample CSV format for batch prediction |

---

## ⚙️ Setup Instructions

### 1. Create a Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📝 Input Features

| Feature        | Description                                    |
| -------------- | ---------------------------------------------- |
| `LotArea`      | Lot size in square feet                        |
| `YearBuilt`    | Year the house was built                       |
| `OverallQual`  | Overall material and finish quality (1–10)     |
| `OverallCond`  | Overall condition of the house (1–10)          |
| `GrLivArea`    | Above ground living area in square feet        |
| `GarageCars`   | Garage capacity in number of cars              |
| `GarageArea`   | Size of the garage in square feet              |
| `FullBath`     | Number of full bathrooms                       |
| `TotRmsAbvGrd` | Total rooms above ground (excluding bathrooms) |

---

## 📁 Batch Prediction

Upload a .csv file with columns like:
```bash
LotArea,YearBuilt,OverallQual,OverallCond,GrLivArea,GarageCars,GarageArea,FullBath,TotRmsAbvGrd
8450,2003,7,5,1710,2,548,2,8
9600,1976,6,8,1262,2,460,2,6
...
```
You'll receive predictions in the app, and also get an option to download the result as a CSV.

---

## 🔍 Example Output

- 💰 Price Estimate: $256,840.23
- 📏 Confidence Interval: $247,101.03 – $266,452.11
- 📍 Location Map
- 📊 Radar Chart (vs. average)
- ✅ Category breakdown (Low, Medium, High per feature)

---

## 📦 Requirements

See requirements.txt, but core dependencies include:

- streamlit
- pandas
- numpy
- xgboost
- scikit-learn
- matplotlib
- plotly
- joblib

---

## 📌 Demo

<img width="1920" height="950" alt="prediction" src="https://github.com/user-attachments/assets/57c48322-55fb-4000-b9bc-2fd807f1e092" />

<img width="1920" height="946" alt="map" src="https://github.com/user-attachments/assets/17d5a878-9495-4a52-ab28-c0e540d14a72" />

<img width="1920" height="950" alt="radar chart" src="https://github.com/user-attachments/assets/ae04d667-bedb-45e0-947e-7ad66f6e8498" />

<img width="1920" height="943" alt="analysis" src="https://github.com/user-attachments/assets/cad295b0-7994-4a4f-bd15-0cd285075825" />




