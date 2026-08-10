import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt


# Page settings
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# Load trained model
with open(
    "student_model.pkl",
    "rb"
) as file:

    model = pickle.load(file)


# Title
st.title(
    "🎓 AI-Driven Student Performance Prediction System"
)

st.write(
    "Predict a student's expected final marks "
    "using Machine Learning."
)

st.divider()


# Student input
st.subheader("📊 Enter Student Details")


col1, col2 = st.columns(2)


with col1:

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0
    )

    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=12.0,
        value=4.0
    )

    previous_marks = st.number_input(
        "Previous Exam Marks",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )


with col2:

    assignment_marks = st.number_input(
        "Assignment Marks",
        min_value=0.0,
        max_value=100.0,
        value=75.0
    )

    internal_marks = st.number_input(
        "Internal Assessment Marks",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )


# Prediction button
if st.button("🔮 Predict Performance"):

    input_data = pd.DataFrame({
        "Attendance": [attendance],
        "Study_Hours": [study_hours],
        "Previous_Marks": [previous_marks],
        "Assignment_Marks": [assignment_marks],
        "Internal_Marks": [internal_marks]
    })


    # Prediction
    prediction = model.predict(
        input_data
    )[0]


    # Keep between 0 and 100
    prediction = max(
        0,
        min(100, prediction)
    )


    # Performance category
    if prediction >= 85:

        performance = "Excellent 🌟"

    elif prediction >= 70:

        performance = "Good 👍"

    elif prediction >= 50:

        performance = "Average 🙂"

    else:

        performance = "Needs Improvement 📚"


    st.divider()

    st.subheader("🎯 Prediction Result")


    result1, result2 = st.columns(2)


    with result1:

        st.metric(
            "Predicted Final Marks",
            f"{prediction:.2f} / 100"
        )


    with result2:

        st.metric(
            "Performance",
            performance
        )


    # Progress bar
    st.write("### Performance Score")

    st.progress(
        int(prediction)
    )


    # Student details
    st.write("### Student Information")

    st.dataframe(
        input_data,
        use_container_width=True
    )


    # Graph
    st.write(
        "### Performance Factors"
    )


    categories = [
        "Attendance",
        "Previous Marks",
        "Assignment",
        "Internal"
    ]


    values = [
        attendance,
        previous_marks,
        assignment_marks,
        internal_marks
    ]


    fig, ax = plt.subplots()

    ax.bar(
        categories,
        values
    )

    ax.set_ylim(
        0,
        100
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Student Performance Factors"
    )

    plt.xticks(
        rotation=20
    )

    st.pyplot(fig)


st.divider()

st.caption(
    "AI-Driven Student Performance Prediction System"
)