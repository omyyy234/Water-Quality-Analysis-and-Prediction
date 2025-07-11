
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# ---- Page Setup ----
st.set_page_config(page_title="Water Quality Web App", layout="wide")
st.title("💧 Water Quality Analysis & Prediction Web App")

# ---- Load Data ----
@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")
    return df

data = load_data()

# ---- Sidebar Navigation ----
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Data Overview", "Visualization", "Model Training", "Predict Potability"])

# ---- Data Overview ----
if app_mode == "Data Overview":
    st.subheader("Dataset Preview")
    st.dataframe(data.head())
    st.markdown("**Shape:** " + str(data.shape))
    st.markdown("**Missing Values:**")
    st.write(data.isnull().sum())

# ---- Visualization ----
elif app_mode == "Visualization":
    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Select feature", data.columns[:-1])
    fig = ff.create_distplot([data[selected_feature].dropna()], group_labels=[selected_feature])
    st.plotly_chart(fig)

    st.subheader("Potability Pie Chart")
    potability_counts = data['Potability'].value_counts().reset_index()
    potability_counts.columns = ['Potability', 'Count']
    potability_counts['Potability'] = potability_counts['Potability'].map({0: 'Not Potable', 1: 'Potable'})
    fig2 = px.pie(potability_counts, names='Potability', values='Count', template='plotly_dark')
    st.plotly_chart(fig2)

# ---- Model Training ----
elif app_mode == "Model Training":
    st.subheader("Train Machine Learning Models")

    df = data.copy()
    df.fillna(df.median(), inplace=True)
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        st.write(f"**{name} Accuracy:** {acc:.2f}")

    best_model_name = max(results, key=results.get)
    st.success(f"✅ Best Model: {best_model_name} ({results[best_model_name]:.2f})")

# ---- Predict Potability ----
elif app_mode == "Predict Potability":
    st.subheader("Enter Parameters to Predict Potability")

    input_data = {}
    for col in data.columns[:-1]:
        input_data[col] = st.number_input(f"{col}", min_value=0.0, step=0.1)

    input_df = pd.DataFrame([input_data])
    input_scaled = StandardScaler().fit(data.drop("Potability", axis=1)).transform(input_df)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(StandardScaler().fit_transform(data.drop("Potability", axis=1)), data["Potability"])
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("✅ This water sample is likely **Potable**.")
    else:
        st.error("⚠️ This water sample is likely **Not Potable**.")
