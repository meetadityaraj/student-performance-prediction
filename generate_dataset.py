import pandas as pd
import numpy as np

np.random.seed(42)

number_of_students = 200

attendance = np.random.randint(50, 101, number_of_students)

study_hours = np.round(
    np.random.uniform(1, 8, number_of_students), 1
)

previous_marks = np.random.randint(35, 96, number_of_students)

assignment_marks = np.random.randint(40, 101, number_of_students)

internal_marks = np.random.randint(40, 96, number_of_students)

final_marks = (
    0.20 * attendance
    + 2.5 * study_hours
    + 0.35 * previous_marks
    + 0.15 * assignment_marks
    + 0.30 * internal_marks
)

final_marks = final_marks + np.random.normal(
    0, 3, number_of_students
)

final_marks = np.clip(final_marks, 0, 100)

final_marks = np.round(final_marks, 2)

data = pd.DataFrame({
    "Attendance": attendance,
    "Study_Hours": study_hours,
    "Previous_Marks": previous_marks,
    "Assignment_Marks": assignment_marks,
    "Internal_Marks": internal_marks,
    "Final_Marks": final_marks
})

data.to_csv(
    "student_performance.csv",
    index=False
)

print("Dataset created successfully!")
print(data.head())