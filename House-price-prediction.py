# Generated from: House-price-prediction.ipynb
# Converted at: 2026-09-27T16:31:01.211Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

df = pd.read_csv("customer_data_preprocessing.csv")
df.head()

df.info()

df.isnull().sum()

df.dtypes

df.duplicated().sum()

df = pd.read_csv("customer_data_preprocessing.csv")

education_map = {
    'High School': 0,
    'Bachelors': 1,
    'Masters': 2,
    'PhD': 3
}

df['education_level'] = df['education_level'].map(education_map)

df['education_level'] = df['education_level'].fillna(
    df['education_level'].median()
)

rating_map = {
    'Very Low': 0,
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Very High': 4
}

df['satisfaction_rating'] = df['satisfaction_rating'].map(rating_map)

df['satisfaction_rating'] = df['satisfaction_rating'].fillna(
    df['satisfaction_rating'].median()
)

print(df[['education_level', 'satisfaction_rating']].head(10))

print(df[['education_level', 'satisfaction_rating']].head())
print(df[['education_level', 'satisfaction_rating']].isnull().sum())

df = pd.get_dummies(df, columns=['gender'], dtype=int)

df.head()

df = pd.get_dummies(df, columns=['city'], dtype=int)
df.head()

df = pd.get_dummies(df, columns=['product_category'], dtype=int)
df.head()

import pandas as pd
import numpy as np

df = pd.read_csv("customer_data_preprocessing.csv")

print(df.shape)

df['is_active'] = df['is_active'].map({
    'Yes': 1,
    'No': 0
})

df['churn'] = df['churn'].map({
    'Yes': 1,
    'No': 0
})

df['is_active'] = df['is_active'].fillna(df['is_active'].mode()[0])
df['churn'] = df['churn'].fillna(df['churn'].mode()[0])

print(df[['is_active', 'churn']].head())
print(df[['is_active', 'churn']].isnull().sum())

df['is_active'].corr(df['churn'])

numeric_cols = [
    'annual_income',
    'purchase_amount',
    'credit_score',
    'membership_years',
    'num_transactions',
    'satisfaction_score'
]

df[numeric_cols] = df[numeric_cols].fillna(
    df[numeric_cols].median()
)

df['education_level'] = df['education_level'].fillna(
    df['education_level'].median()
)

df['satisfaction_rating'] = df['satisfaction_rating'].fillna(
    df['satisfaction_rating'].median()
)

df['is_active'] = df['is_active'].fillna(df['is_active'].mode()[0])
df['churn'] = df['churn'].fillna(df['churn'].mode()[0])


import pip
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
df.head()

correlation = df.select_dtypes(include='number').corr()
correlation

df = df.drop(columns=['monthly_income'])

df['churn'].value_counts()
df['churn'].unique()

X = df.drop('churn', axis=1)
y = df['churn']

print(X.shape)
print(y.shape)


print(df.columns.tolist())

X = df.drop(columns=['churn'])
y = df['churn']

print("X:", X.shape)
print("y:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

from sklearn.model_selection import train_test_split

X = df.drop(columns=['churn'])
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)

print(X_train.dtypes)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Remove ID and name
X = df.drop(columns=['churn', 'customer_id', 'full_name'])

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'string']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

print("Categorical columns:", list(categorical_cols))
print("Numerical columns:", list(numerical_cols))

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Convert dates to datetime
df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')

# Create useful features from dates
df['signup_year'] = df['signup_date'].dt.year
df['signup_month'] = df['signup_date'].dt.month

# Calculate age
df['age'] = (pd.Timestamp.today() - df['date_of_birth']).dt.days / 365.25

# Drop original dates
df = df.drop(columns=['signup_date', 'date_of_birth'])

print(df.shape)
print(df.dtypes)

X = df.drop(columns=['churn', 'customer_id', 'full_name'])
y = df['churn']

print("X:", X.shape)
print("y:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_cols = X_train.select_dtypes(include=['object', 'string']).columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

print("Categorical:", list(categorical_cols))
print("Numerical:", list(numerical_cols))

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("X_train processed:", X_train_processed.shape)
print("X_test processed:", X_test_processed.shape)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("X_train processed:", X_train_processed.shape)
print("X_test processed:", X_test_processed.shape)

from sklearn.neural_network import MLPClassifier

nn_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    max_iter=500,
    random_state=42
)

nn_model.fit(X_train_processed, y_train)

y_pred_nn = nn_model.predict(X_test_processed)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Neural Network Results")
print("Accuracy:", accuracy_score(y_test, y_pred_nn))
print("Precision:", precision_score(y_test, y_pred_nn))
print("Recall:", recall_score(y_test, y_pred_nn))
print("F1-score:", f1_score(y_test, y_pred_nn))

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train_processed, y_train)

y_pred_rf = rf_model.predict(X_test_processed)

print("Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1-score:", f1_score(y_test, y_pred_rf))

from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train_processed, y_train)

y_pred_xgb = xgb_model.predict(X_test_processed)

print("XGBoost Results")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("F1-score:", f1_score(y_test, y_pred_xgb))

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

models = ['Neural Network', 'Random Forest', 'XGBoost']
accuracy = [0.5425, 0.6151, 0.6268]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
plt.show()

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']

nn_scores = [0.5425, 0.5389, 0.5569, 0.5478]
rf_scores = [0.6151, 0.6187, 0.5903, 0.6042]
xgb_scores = [0.6268, 0.6256, 0.6225, 0.6241]

x = range(len(metrics))

plt.figure(figsize=(10, 6))

plt.plot(x, nn_scores, marker='o', label='Neural Network')
plt.plot(x, rf_scores, marker='o', label='Random Forest')
plt.plot(x, xgb_scores, marker='o', label='XGBoost')

plt.xticks(list(x), metrics)
plt.ylabel('Score')
plt.xlabel('Evaluation Metric')
plt.title('Performance Comparison of Machine Learning Models')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_xgb
)

plt.title('XGBoost Confusion Matrix')
plt.show()