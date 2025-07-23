import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Salary Predictor with SHAP", layout="wide")

st.title("💼 Salary Prediction with Explainability & Fairness")

df = pd.read_csv('data/salary_dataset.csv')
df.dropna(inplace=True)

df['Gender'] = df['Gender'].replace({'Female': 0, 'Male': 1, 'Other': 2}).infer_objects(copy=False)
df['Education Level'] = df['Education Level'].str.strip().str.lower()
df['Education Level'] = df['Education Level'].replace({
    "high school": "high school",
    "bachelor's degree": "bachelor's",
    "master's degree": "master's",
    "masters": "master's",
    "phd": "phd",
    "ph.d": "phd",
    "ph.d.": "phd"
})
edu_map = {"high school": 0, "bachelor's": 1, "master's": 2, "phd": 3}
df['Education Level'] = df['Education Level'].map(edu_map)

if 'Job Title' in df.columns:
    job_title_counts = df['Job Title'].value_counts()
    rare_titles = job_title_counts[job_title_counts <= 25].index
    df['Job Title'] = df['Job Title'].apply(lambda x: 'Others' if x in rare_titles else x)
    dummies = pd.get_dummies(df['Job Title'], drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df.drop('Job Title', axis=1, inplace=True)

features = df.drop('Salary', axis=1)
target = df['Salary']
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)

model = RandomForestRegressor(n_estimators=20)
model.fit(x_train, y_train)

st.header("📊 Predict Salary")
with st.form("predict_form"):
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    edu = st.selectbox("Education Level", list(edu_map.keys()))
    exp = st.slider("Years of Experience", 0, 50, 2)
    
    job_cols = [col for col in x_train.columns if col not in ['Age', 'Gender', 'Education Level', 'Years of Experience']]
    job_input = st.selectbox("Job Title", ['Others'] + [col for col in job_cols if col != 'Others'])
    
    submit = st.form_submit_button("Predict")

if submit:
    input_dict = {
        'Age': age,
        'Gender': {'Male': 1, 'Female': 0, 'Other': 2}[gender],
        'Education Level': edu_map[edu],
        'Years of Experience': exp
    }

    for col in job_cols:
        input_dict[col] = 1 if col == job_input else 0

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    st.success(f"🧾 Predicted Salary: ${prediction:,.2f}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)

    st.subheader("📌 Feature Importance (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.summary_plot(shap_values, x_test, plot_type='bar', show=False)
    st.pyplot(fig)

    x_test_gender = x_test.copy()
    x_test_gender['Gender'] = x_test_gender['Gender'].replace({0: 'Female', 1: 'Male', 2: 'Other'})
    x_test_gender['Actual'] = y_test.values
    x_test_gender['Predicted'] = model.predict(x_test)

    mean_pred_salary = x_test_gender.groupby('Gender')['Predicted'].mean()
    dir_fm = mean_pred_salary.get('Female', 0) / mean_pred_salary.get('Male', 1)
    dir_of = mean_pred_salary.get('Other', 0) / mean_pred_salary.get('Male', 1)

    st.subheader("⚖️ Disparate Impact Ratio")
    st.write(f"**Female/Male:** {dir_fm:.2f}")
    st.write(f"**Other/Male:** {dir_of:.2f}")
