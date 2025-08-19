import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso

# --- CORRECTED: Load and Preprocess Data ---
try:
    df1 = pd.read_csv('gym_members_exercise_tracking.csv')
    df2 = pd.read_csv('exercise_dataset.csv')
except FileNotFoundError:
    print("Error: Make sure 'gym_members_exercise_tracking.csv' and 'exercise_dataset.csv' are in the same directory.")
    exit()

# Process the first dataframe (df1)
df1_processed = df1[['Age', 'Gender', 'Weight (kg)', 'Session_Duration (hours)', 'Avg_BPM', 'Calories_Burned']].copy()

# Process the second dataframe (df2)
df2_processed = df2.rename(columns={
    'Actual Weight': 'Weight (kg)', 'Duration': 'Session_Duration (hours)',
    'Heart Rate': 'Avg_BPM', 'Calories Burn': 'Calories_Burned'
})
# Only convert the duration from minutes to hours.
df2_processed['Session_Duration (hours)'] /= 60
df2_processed = df2_processed[['Age', 'Gender', 'Weight (kg)', 'Session_Duration (hours)', 'Avg_BPM', 'Calories_Burned']]

# Combine into a single DataFrame and clean it
combined_df = pd.concat([df1_processed, df2_processed], ignore_index=True).dropna()
combined_df = combined_df[combined_df['Calories_Burned'] > 0]
# --- End of Data Preprocessing ---


# --- Feature Engineering (on ALL data) ---
X_numerical = combined_df[['Age', 'Weight (kg)', 'Session_Duration (hours)', 'Avg_BPM']]
y = combined_df['Calories_Burned']
X_gender_encoded = pd.get_dummies(combined_df[['Gender']], drop_first=True)
X_original = pd.concat([X_numerical, X_gender_encoded], axis=1)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_original)
poly_feature_names = poly.get_feature_names_out(X_original.columns)
X_engineered = pd.DataFrame(X_poly, columns=poly_feature_names)

# --- Scale ALL Engineered Data ---
scaler = StandardScaler()
X_engineered_scaled = scaler.fit_transform(X_engineered)

# --- Train the Final Lasso Model ---
# Use the best alpha found during cross-validation from the previous run
final_alpha = 0.2396
final_model = Lasso(alpha=final_alpha, max_iter=10000, random_state=42)
final_model.fit(X_engineered_scaled, y)

# --- 1. Print Coefficients for App ---
print("--- Final Application Model Coefficients ---")
print("# These coefficients are based on SCALED data.")
active_coeffs = []
for feature, coef in zip(X_engineered.columns, final_model.coef_):
    if abs(coef) > 1e-5:
        active_coeffs.append(f"'{feature}': {coef:.4f},")
print("\n".join(active_coeffs))
print(f"'intercept': {final_model.intercept_:.4f}")
print("------------------------------------------\n")

# --- 2. Print Scaler Parameters for App ---
print("--- Scaler Parameters (mean_ and scale_) ---")
print("# These are needed to scale new data in the app.")
scaler_means = []
for feature, mean in zip(X_engineered.columns, scaler.mean_):
     scaler_means.append(f"'{feature}': {mean:.4f},")

scaler_scales = []
for feature, scale in zip(X_engineered.columns, scaler.scale_):
     scaler_scales.append(f"'{feature}': {scale:.4f},")

print("\nSCALER_MEANS = {")
print("\n".join(scaler_means))
print("}")

print("\nSCALER_SCALES = {")
print("\n".join(scaler_scales))
print("}")
print("--------------------------------------------")
