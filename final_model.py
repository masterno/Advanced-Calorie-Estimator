import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load, preprocess, and combine data (same as before)
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

# Perform feature engineering (same as before)
X_numerical = combined_df[['Age', 'Weight (kg)', 'Session_Duration (hours)', 'Avg_BPM']]
y = combined_df['Calories_Burned']
X_gender_encoded = pd.get_dummies(combined_df[['Gender']], drop_first=True)
X_original = pd.concat([X_numerical, X_gender_encoded], axis=1)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_original)
poly_feature_names = poly.get_feature_names_out(X_original.columns)
X_engineered = pd.DataFrame(X_poly, columns=poly_feature_names)

# Train the final Linear Regression model on ALL engineered data
final_app_model = LinearRegression()
final_app_model.fit(X_engineered, y)

# Print the coefficients needed for the app
print("--- Final App Model Coefficients ---")
for feature, coef in zip(X_engineered.columns, final_app_model.coef_):
    print(f"'{feature}': {coef:.4f},")
print(f"'intercept': {final_app_model.intercept_:.4f}")
print("------------------------------------")
