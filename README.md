🚀 Advanced Calorie Estimator
Welcome to the Advanced Calorie Estimator! This web application provides a personalized estimate of calories burned during an exercise session. Unlike simple calculators, this tool is powered by a machine learning model trained on thousands of real-world workout data points.

How to Use the App
Using the estimator is straightforward:

Enter Your Details: Fill in the five required fields:

Age: Your age in years.

Gender: Select Male or Female.

Weight (kg): Your current weight in kilograms.

Session Duration (hours): The total length of your workout in hours (e.g., enter 1.5 for 90 minutes).

Average Heart Rate (BPM): Your average heart rate during the session, in beats per minute.

Calculate: Click the "Calculate Calories" button.

View Result: Your estimated calories burned will appear at the bottom of the form.

The Data Science Journey
This model is the result of a detailed data science process designed to create a robust and reliable prediction tool.

The Goal
The primary goal was to build a model that could accurately predict the number of calories burned during a workout using simple, measurable inputs, moving beyond basic linear formulas to capture more complex physiological relationships.

Data & Cleaning
The model was trained on a combined dataset from two sources on Kaggle, totaling nearly 5,000 unique exercise sessions.

A critical step in the process was data cleaning. We discovered that the two datasets had inconsistent column names and units for workout duration (minutes vs. hours). The most important finding was ensuring all data was standardized to the same units before training. Initial attempts to train on mixed-unit data led to a model that appeared accurate but was actually learning from the data errors—a classic "garbage in, garbage out" scenario. Correcting these inconsistencies was crucial for building a trustworthy model.

Feature Engineering: Beyond the Basics
A simple linear model can only learn straight-line relationships. To capture the complex, non-linear nature of human physiology, we used feature engineering. We created new features by combining the original inputs, including:

Squared terms (e.g., Age^2, Weight (kg)^2): To allow the model to learn curved relationships.

Interaction terms (e.g., Weight (kg) * Avg_BPM): To let the model learn that the effect of one feature can depend on another.

This expanded our feature set from 5 initial inputs to 20, giving the model much more information to learn from.

Model Selection: Why Lasso Regression?
We initially explored a standard LinearRegression model, but with so many new features, it was at high risk of overfitting—learning the noise in the data rather than the true patterns.

To solve this, we upgraded to a Lasso (L1 Regularization) model. Lasso works by penalizing complex models and can shrink the coefficients of less important features all the way to zero, effectively performing automatic feature selection. This resulted in a simpler, more robust model that is better at generalizing to new, unseen data.

Key Insights from the Model
The final model produced some fascinating insights that go beyond simple formulas:

The Power of Interaction: The most significant factor in calorie burn isn't just how long you work out or how high your heart rate is, but the combination of both. The model found a strong positive interaction between Session_Duration and Avg_BPM, meaning that maintaining a high heart rate for a longer duration has an exponential, rather than an additive, effect on calories burned.

The Non-Linear Effect of Weight: The relationship between weight and calorie burn isn't a straight line. The model learned a curved relationship, where calorie burn increases at an accelerating rate for heavier individuals. This reflects the greater energy expenditure required to move more body mass.

Diminishing Returns of Heart Rate: Interestingly, the model found that the effect of heart rate has diminishing returns. While increasing your heart rate from low to moderate burns a lot of calories, the additional calories burned per beat starts to level off at very high intensities. This suggests a peak efficiency zone for calorie expenditure during a workout.

Honest Performance: After cleaning the data, the model achieved an R-squared of ~0.75. This means it can explain about 75% of the variation in calories burned, which is a strong and, more importantly, honest measure of its performance on clean, reliable data.

Handling Model Limitations: Extrapolation Safeguard
A key discovery during testing was that the model's training data had a maximum workout duration of 2 hours. When asked to predict for longer durations (e.g., 3 or 4 hours), the model would extrapolate its learned exponential curve, leading to physiologically impossible calorie estimates.

To solve this, the app includes a dynamic safeguard:

For workouts up to 2 hours, it uses the machine learning model directly.

For workouts longer than 2 hours, it first calculates a baseline calorie burn at the 2-hour mark. It then uses this to create a personalized, linear burn rate (with a fatigue factor applied) to estimate the calories for the remaining duration. This prevents unrealistic predictions and makes the app more robust.

The final web app uses this trained Lasso model, along with the specific scaling parameters and safeguards from the dataset, to make its predictions.