import pandas as pd

# Load, preprocess, and combine data (same as before)
try:
    df1 = pd.read_csv('gym_members_exercise_tracking.csv')
    df2 = pd.read_csv('exercise_dataset.csv')
except FileNotFoundError:
    print("Make sure 'gym_members_exercise_tracking.csv' and 'exercise_dataset.csv' are in the same directory.")
    exit()


# Process the first dataframe
df1_processed = df1[['Age', 'Gender', 'Weight (kg)', 'Session_Duration (hours)', 'Avg_BPM', 'Calories_Burned']].copy()

# Process the second dataframe
df2_processed = df2.rename(columns={
    'Actual Weight': 'Weight (kg)', 'Duration': 'Session_Duration (hours)',
    'Heart Rate': 'Avg_BPM', 'Calories Burn': 'Calories_Burned'
})
df2_processed['Weight (kg)'] *= 0.453592
df2_processed['Session_Duration (hours)'] /= 60
df2_processed = df2_processed[['Age', 'Gender', 'Weight (kg)', 'Session_Duration (hours)', 'Avg_BPM', 'Calories_Burned']]

# Combine into a single DataFrame
combined_df = pd.concat([df1_processed, df2_processed], ignore_index=True).dropna()
combined_df = combined_df[combined_df['Calories_Burned'] > 0]


# --- Calculate and Print Statistics ---
mean_calories = combined_df['Calories_Burned'].mean()
std_calories = combined_df['Calories_Burned'].std()

print("--- Descriptive Statistics for Calories Burned ---")
print(f"Mean: {mean_calories:.2f} calories")
print(f"Standard Deviation: {std_calories:.2f} calories")
print("-------------------------------------------------")
print(f"\nInterpretation: A typical value is around {mean_calories:.0f} calories, and most values fall within {std_calories:.0f} calories of that mean.")

