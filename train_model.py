import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Load dataset
data = pd.read_csv("student_performance.csv")

print("Dataset loaded successfully!")


# Input features
X = data[
    [
        "Attendance",
        "Study_Hours",
        "Previous_Marks",
        "Assignment_Marks",
        "Internal_Marks"
    ]
]


# Target variable
y = data["Final_Marks"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Create model
model = LinearRegression()


# Train model
model.fit(X_train, y_train)

print("Model trained successfully!")


# Predictions
predictions = model.predict(X_test)


# Model evaluation
mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\nModel Performance")
print("-------------------------")

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 2))


# Save model
with open(
    "student_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


print("\nModel saved successfully!")