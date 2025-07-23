💼 Employee Salary Prediction with Explainability & Fairness Analysis

This project uses machine learning to predict employee salaries based on features such as education, experience, gender, and job title. It also incorporates **explainability** (via SHAP) and **fairness analysis** across gender groups.

📊 Features

- Predict salaries using Random Forest Regressor
- SHAP-based model explainability
- Gender-based fairness analysis
- Streamlit web app interface
- Auto-generated and saved SHAP plots

🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python (Pandas, scikit-learn)
- **Explainability**: SHAP (SHapley Additive Explanations)
- **Visualization**: Matplotlib, Seaborn
- **Fairness Metrics**: MAE by Gender, Disparate Impact Ratio

🗂️ Project Structure

├── app.py # Streamlit App
├── Emp_Sal_Predict.ipynb # Notebook with core logic
├── requirements.txt # Dependencies
├── README.md # Project Documentation
├── images/ # Saved SHAP plots
│ ├── shap_summary_bar.png
│ ├── shap_summary_dot.png
│ └── shap_waterfall.png


🚀 Run the App

```bash
pip install -r requirements.txt
streamlit run app.py

📈 SHAP Explainability

Key SHAP plots are auto-generated in the images/ directory:

shap_summary_bar.png

shap_summary_dot.png

shap_waterfall.png

⚖️ Fairness Analysis

The app computes:

MAE by Gender

Disparate Impact Ratio (Female/Male and Other/Male)

📄 Dataset

This project uses a CSV dataset containing employee salary details and features like:

Gender

Education Level

Years of Experience

Job Title

MIT License

Copyright (c) 2025 Akhil V Nair

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



