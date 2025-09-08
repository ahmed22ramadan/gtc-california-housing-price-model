
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

st.title("🏡 California Housing Price Prediction")

# تحميل البيانات
@st.cache_data
def load_data():
    from sklearn.datasets import fetch_california_housing
    cali = fetch_california_housing(as_frame=True)
    return cali

cali = load_data()
df = pd.DataFrame(cali.data, columns=cali.feature_names)
df['MedHouseVal'] = cali.target

st.subheader("Dataset Preview")
st.dataframe(df.head())

# اختيار مميزات للتدريب
features = st.multiselect("اختر الخصائص:", df.columns[:-1].tolist(), default=df.columns[:5].tolist())

X = df[features]
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# اختيار الموديل
model_choice = st.selectbox("اختر الموديل:", ["Linear Regression", "Random Forest", "XGBoost"])

if model_choice == "Linear Regression":
    model = LinearRegression()
elif model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
else:
    model = XGBRegressor(
        n_estimators=50, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42
    )

# تدريب الموديل
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# عرض النتائج
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.success(f"Mean Squared Error (MSE): {mse:.4f}")
st.success(f"R² Score: {r2:.4f}")

# رسم مقارنة التوقعات مع القيم الحقيقية
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y_test, y_pred, alpha=0.3)
ax.set_xlabel("Actual Prices")
ax.set_ylabel("Predicted Prices")
ax.set_title("Actual vs Predicted Prices")
st.pyplot(fig)
