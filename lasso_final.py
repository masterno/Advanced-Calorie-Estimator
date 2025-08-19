import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso

# --- Load and Preprocess Data ---
try:
    df1 = pd.read_csv('gym_members_exercise_tracking.csv')
    df2 = pd.read_csv('exercise_dataset.csv')
except FileNotFoundError:
    print("Make sure 'gym_members_exercise_tracking.csv' and 'exercise_dataset.csv' are in the same directory.")
    exit()

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
# Note: We fit and transform the entire dataset now
scaler = StandardScaler()
X_engineered_scaled = scaler.fit_transform(X_engineered)

# --- Train the Final Lasso Model ---
# Use the best alpha found during cross-validation
final_alpha = 0.2471
final_model = Lasso(alpha=final_alpha, max_iter=10000, random_state=42)
final_model.fit(X_engineered_scaled, y)

# --- Print Coefficients for App ---
# These are the values you'll need for your application's prediction function
print("--- Final Application Model Coefficients ---")
print("# These coefficients are based on SCALED data.")
for feature, coef in zip(X_engineered.columns, final_model.coef_):
    # Only print features the model decided to keep
    if abs(coef) > 0.0001:
        print(f"'{feature}': {coef:.4f},")

print(f"\n'intercept': {final_model.intercept_:.4f}")
print("------------------------------------------")
