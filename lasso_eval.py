import pandas as pd
import numpy as np  # <-- Add this import
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- START: DATA LOADING AND PREPROCESSING ---
df1 = pd.read_csv('gym_members_exercise_tracking.csv')
df2 = pd.read_csv('exercise_dataset.csv')

df1_processed = df1[['Age', 'Gender', 'Weight (kg)', 'Session_Duration (hours)', 'Avg_BPM', 'Calories_Burned']].copy()

df2_processed = df2.rename(columns={
    'Actual Weight': 'Weight (kg)', 'Duration': 'Session_Duration (hours)',
    'Heart Rate': 'Avg_BPM', 'Calories Burn': 'Calories_Burned'
})
df2_processed['Weight (kg)'] *= 0.453592
df2_processed['Session_Duration (hours)'] /= 60
df2_processed = df2_processed[['Age', 'Gender', 'Weight (kg)', 'Session_Duration (hours)', 'Avg_BPM', 'Calories_Burned']]

combined_df = pd.concat([df1_processed, df2_processed], ignore_index=True).dropna()
combined_df = combined_df[combined_df['Calories_Burned'] > 0]
# --- END: DATA LOADING AND PREPROCESSING ---


# --- FEATURE ENGINEERING ---
X_numerical = combined_df[['Age', 'Weight (kg)', 'Session_Duration (hours)', 'Avg_BPM']]
y = combined_df['Calories_Burned']
X_gender_encoded = pd.get_dummies(combined_df[['Gender']], drop_first=True)
X_original = pd.concat([X_numerical, X_gender_encoded], axis=1)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_original)
poly_feature_names = poly.get_feature_names_out(X_original.columns)
X_engineered = pd.DataFrame(X_poly, columns=poly_feature_names)

X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso_cv_model = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv_model.fit(X_train_scaled, y_train)

# --- START: CORRECTED MODEL EVALUATION ---
train_score = lasso_cv_model.score(X_train_scaled, y_train)
test_score = lasso_cv_model.score(X_test_scaled, y_test)
y_pred = lasso_cv_model.predict(X_test_scaled)

# Calculate MSE first, then take the square root for RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
# --- END: CORRECTED MODEL EVALUATION ---


print("--- LassoCV Model Performance ---")
print(f"Best Alpha found by Cross-Validation: {lasso_cv_model.alpha_:.4f}")
print(f"R-squared on Training Set: {train_score:.4f}")
print(f"R-squared on Test Set: {test_score:.4f}")
print(f"Root Mean Squared Error on Test Set: {rmse:.4f}")
print("---------------------------------")


print("\n--- Features Kept by Lasso Model ---")
retained_features = []
for feature, coef in zip(X_engineered.columns, lasso_cv_model.coef_):
    if abs(coef) > 0.0001:
        print(f"'{feature}': {coef:.4f}")
        retained_features.append(feature)

print(f"\nLasso selected {len(retained_features)} features out of {len(X_engineered.columns)} total.")
print("------------------------------------")