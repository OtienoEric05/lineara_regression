import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib  # for saving the model

# Load the CSV file
df = pd.read_csv('road_accident.csv')

# Preview the first few rows of the dataset
print(df.head())

# Ensure there are no missing values in the relevant columns
df = df.dropna(subset=['Accident ID', 'Severity', 'Road Type', 'Weather', 'Vehicle Type', 'Speed', 'Driver Age', 'Driver Experience', 'Accident Type'])

# Use one-hot encoding for categorical variables
df = pd.get_dummies(df, columns=['Road Type', 'Weather', 'Vehicle Type', 'Accident Type'], drop_first=True)

# Define dependent variable (y) and independent variables (X)
X = df.drop(columns=['Accident ID', 'Severity'])
y = df['Severity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on test set: {mse:.2f}")

# Save the model
joblib.dump(model, 'accident_severity_model.pkl')
print("Model saved as 'accident_severity_model.pkl'")

# Preparing new accident data for prediction
new_data = {
    'Speed': 50,
    'Driver Age': 30,
    'Driver Experience': 10,
    'Road Type_City Street': 0,
    'Road Type_Highway': 1,  # Assuming the new accident is on a Highway
    'Road Type_Rural Road': 0,
    'Weather_Rainy': 0,
    'Weather_Snowy': 0,
    'Weather_Sunny': 1,
    'Vehicle Type_Car': 1,
    'Vehicle Type_Truck': 0,
    'Vehicle Type_Motorcycle': 0,
    'Accident Type_Rear-end collision': 1,
    'Accident Type_Side-swipe collision': 0,
    'Accident Type_Single-vehicle rollover': 0
}

# Convert the new data into a DataFrame
new_accident = pd.DataFrame(new_data, index=[0])

# Ensure the columns of new_accident match those of the trained model
new_accident = new_accident.reindex(columns=X.columns, fill_value=0)

# Predict the accident severity for the new data
severity_prediction = model.predict(new_accident)
print(f"Predicted Accident Severity for the new case: {severity_prediction[0]:.2f}")
