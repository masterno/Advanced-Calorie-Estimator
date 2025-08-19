import pandas as pd

# --- Load and Preprocess Data ---
# This section is the same as your previous scripts to ensure consistency.
try:
    df1 = pd.read_csv('gym_members_exercise_tracking.csv')
    df2 = pd.read_csv('exercise_dataset.csv')
except FileNotFoundError:
    print("Error: Make sure 'gym_members_exercise_tracking.csv' and 'exercise_dataset.csv' are in the same directory as this script.")
    exit()

# Process the first dataframe
# The 'Weight (kg)' column is already in kilograms, so no conversion is needed.
df1_processed = df1[['Age', 'Gender', 'Weight (kg)', 'Session_Duration (hours)', 'Avg_BPM', 'Calories_Burned']].copy()


# Process the second dataframe
df2_processed = df2.rename(columns={
    'Actual Weight': 'Weight (kg)', 'Duration': 'Session_Duration (hours)',
    'Heart Rate': 'Avg_BPM', 'Calories Burn': 'Calories_Burned'
})
# --- CORRECTED LOGIC ---
# The 'Actual Weight' column is also in kilograms. NO weight conversion is needed.
# Only convert the duration from minutes to hours.
df2_processed['Session_Duration (hours)'] /= 60
df2_processed = df2_processed[['Age', 'Gender', 'Weight (kg)', 'Session_Duration (hours)', 'Avg_BPM', 'Calories_Burned']]

# Combine into a single DataFrame and clean it
combined_df = pd.concat([df1_processed, df2_processed], ignore_index=True).dropna()
combined_df = combined_df[combined_df['Calories_Burned'] > 0]

# --- Calculate and Print Descriptive Statistics ---

# Select the core features you want to analyze
features_to_analyze = combined_df[['Age', 'Weight (kg)', 'Session_Duration (hours)', 'Avg_BPM']]

# Use the .describe() method to get all the key stats at once
descriptive_stats = features_to_analyze.describe()

print("--- Descriptive Statistics for Key Model Features ---")
print(descriptive_stats)
print("----------------------------------------------------")

# --- Interpretation Guidance ---
print("\n--- How to Interpret These Stats ---")
print("1. 'count': The number of valid data points for each feature.")
print("2. 'mean': The average value. This tells you the central tendency of your data.")
print("3. 'std' (Standard Deviation): How spread out the data is. A large value means a wide range.")
print("4. 'min' / 'max': The minimum and maximum values in your dataset. Your model will be less reliable outside this range.")
print("5. '25%' / '50%' / '75%': The quartiles. 50% of your data lies between the 25% and 75% values.")
print("   - '50%' is also the median value.")
print("--------------------------------------")

