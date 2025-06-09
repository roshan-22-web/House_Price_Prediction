import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

# Load your CSV
df = pd.read_csv("coimbatore_house_prices.csv")

# Preprocess categorical columns consistently
if 'Parking' in df.columns:
    df['Parking'] = df['Parking'].replace({'Yes': 1, 'No': 0})

# Define proper mapping for Age
age_mapping = {'New': 0, 'Moderate': 1, 'Old': 2}
if 'Age' in df.columns:
    df['Age'] = df['Age'].map(age_mapping)

target = 'Price_Lakhs'
X = df.drop(columns=[target])
y = df[target]

# Auto-detect column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Save model and preprocessing information
model_info = {
    'model': pipeline,
    'age_mapping': age_mapping,
    'categorical_cols': categorical_cols
}
joblib.dump(model_info, 'model.pkl')
print("✅ Model trained and saved as model.pkl")