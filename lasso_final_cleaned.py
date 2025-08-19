import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# --- CORRECTED: Load and Preprocess Data ---
try:
    df1 = pd.read_csv('gym_members_exercise_tracking.csv')
    df2 = pd.read_csv('exercise_dataset.csv')
except FileNotFoundError:
    print("Error: Make sure 'gym_members_exercise_tracking.csv' and 'exercise_dataset.csv' are in the same directory.")
    exit()

# Process the first dataframe (df1)
# The 'Weight (kg)' column is already in kilograms, so no conversion is needed.
df1_processed = df1[['Age', 'Gender', 'Weight (kg)', 'Session_Duration (hours)', 'Avg_BPM', 'Calories_Burned']].copy()

# Process the second dataframe (df2)
df2_processed = df2.rename(columns={
    'Actual Weight': 'Weight (kg)', 'Duration': 'Session_Duration (hours)',
    'Heart Rate': 'Avg_BPM', 'Calories Burn': 'Calories_Burned'
})
# The 'Actual Weight' column is also in kilograms. NO weight conversion is needed.
# Only convert the duration from minutes to hours.
df2_processed['Session_Duration (hours)'] /= 60
df2_processed = df2_processed[['Age', 'Gender', 'Weight (kg)', 'Session_Duration (hours)', 'Avg_BPM', 'Calories_Burned']]

# Combine into a single DataFrame and clean it
combined_df = pd.concat([df1_processed, df2_processed], ignore_index=True).dropna()
combined_df = combined_df[combined_df['Calories_Burned'] > 0]
# --- End of Data Preprocessing ---


# --- Feature Engineering ---
X_numerical = combined_df[['Age', 'Weight (kg)', 'Session_Duration (hours)', 'Avg_BPM']]
y = combined_df['Calories_Burned']
X_gender_encoded = pd.get_dummies(combined_df[['Gender']], drop_first=True)
X_original = pd.concat([X_numerical, X_gender_encoded], axis=1)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_original)
poly_feature_names = poly.get_feature_names_out(X_original.columns)
X_engineered = pd.DataFrame(X_poly, columns=poly_feature_names)

# Split the data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Use LassoCV to find the best alpha and train the model
lasso_cv_model = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv_model.fit(X_train_scaled, y_train)

# Evaluate the model
train_score = lasso_cv_model.score(X_train_scaled, y_train)
test_score = lasso_cv_model.score(X_test_scaled, y_test)
y_pred = lasso_cv_model.predict(X_test_scaled)

# Calculate RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("--- LassoCV Model Performance (with corrected data) ---")
print(f"Best Alpha found by Cross-Validation: {lasso_cv_model.alpha_:.4f}")
print(f"R-squared on Training Set: {train_score:.4f}")
print(f"R-squared on Test Set: {test_score:.4f}")
print(f"Root Mean Squared Error on Test Set: {rmse:.4f}")
print("-------------------------------------------------------")

# See which features were kept (non-zero coefficients)
print("\n--- Features Kept by Lasso Model ---")
retained_features = []
for feature, coef in zip(X_engineered.columns, lasso_cv_model.coef_):
    if abs(coef) > 1e-5: # Check if coefficient is not effectively zero
        print(f"'{feature}': {coef:.4f}")
        retained_features.append(feature)

print(f"\nLasso selected {len(retained_features)} features out of {len(X_engineered.columns)} total.")
print("------------------------------------")
