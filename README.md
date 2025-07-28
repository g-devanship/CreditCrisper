# CreditCrisper


A machine learning pipeline for predicting credit default risk using customer demographics, billing behavior, and historical repayment data from a Taiwanese bank.

## 🧠 Project Objective

To predict whether a customer will default on their credit card payment next month, using supervised learning on structured tabular data.

## 💡 Core Ideas

- Credit risk classification based on historical financial behavior
- Exploration of key demographic and transactional variables
- Supervised machine learning with a focus on interpretability and performance
- Gradient boosting for tabular data with minimal preprocessing

## 🛠️ Technologies Used

- **Python**: Data preprocessing, modeling
- **Pandas, NumPy**: Data analysis and manipulation
- **Scikit-learn**: Model evaluation and metrics
- **XGBoost**: Gradient boosting-based classification
- **Matplotlib, Seaborn**: Exploratory data visualization
- **Amazon SageMaker (optional)**: Scalable training

## 🔍 Techniques Applied

- Binary classification: Predict `default.payment.next.month`
- Feature engineering from 25+ input variables (e.g., billing amount, repayment status)
- Handling of categorical features (sex, education, marital status)
- Hyperparameter tuning with `max_depth`, `eta`, `lambda`, etc.
- Evaluation using confusion matrix, precision, recall, F1-score

## 📦 Outcome

- Built an efficient and accurate XGBoost model for financial risk modeling
- Interpreted key contributors to default risk (e.g., bill amount history, late payments)
- Demonstrated the practical application of AI in real-world financial decision-making

---

### 📈 Future Work

- Deploy as an API using Flask or FastAPI
- Integrate with a dashboard for live predictions
- Add SHAP for explainable AI (model interpretability)
