# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Step 1: Load your data
# Load the monarch sightings data
monarch_data = pd.read_csv('/Users/anudeepbonagiri/Downloads/Archive/2023/monarchAdults/fullFile/full_monarch-adult-fall_fall_2023.csv')  # Replace with actual file path

# Load the pesticide usage data with low_memory=False to avoid dtype warnings
pesticide_data = pd.read_csv('/Users/anudeepbonagiri/Downloads/USDA_PDP_AnalyticalResults (1).csv', low_memory=False)  # Replace with actual file path

# Print column names to debug
print("Monarch Data Columns:", monarch_data.columns)
print("Pesticide Data Columns:", pesticide_data.columns)

# Strip whitespace from column names for consistency
monarch_data.columns = monarch_data.columns.str.strip().str.lower()
pesticide_data.columns = pesticide_data.columns.str.strip().str.lower()

# Step 2: Merge the datasets on common columns (e.g., State and Date)
# Ensure the column names used for merging match those in the datasets
merged_data = pd.merge(monarch_data, pesticide_data, left_on=['state', 'date'], right_on=['state', 'date'], how='inner')

# Step 3: Prepare your data for modeling
# Select the feature (pesticide concentration) and target variable (monarch sightings)
X = merged_data[['pesticide_concentration']]  # Independent variable(s)
y = merged_data['monarch_sightings']  # Dependent variable (what you want to predict)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R-squared score

# Print evaluation metrics
print(f'Mean Squared Error: {mse}')
print(f'R-squared score: {r2}')

# Step 8: Visualize the results
plt.scatter(X_test, y_test, color='blue', label='Actual')  # Actual values
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')  # Predicted values
plt.title('Monarch Sightings vs Pesticide Concentration')
plt.xlabel('Pesticide Concentration')
plt.ylabel('Monarch Sightings')
plt.legend()
plt.show()

# Step 9: Optional - Save the model (if you want to use it later)
joblib.dump(model, 'monarch_predictor.pkl')
