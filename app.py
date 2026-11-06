from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "student-performance-processed.csv"
)


st.set_page_config(
    page_title="Student Academic Risk Prediction",
    page_icon="🎓",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


st.title("🎓 Student Academic Risk Prediction System")

st.write(
    "An Early Warning System for identifying students "
    "who may be at academic risk using Machine Learning."
)

st.info(
    "Enter the student's information below "
    "to generate an academic risk prediction."
)


if not MODEL_PATH.exists():
    st.error("Trained model not found.")
    st.stop()

if not DATA_PATH.exists():
    st.error("Processed dataset not found.")
    st.stop()


model = load_model()
df = load_dataset()

X = df.drop(columns=["risk"])

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()


st.subheader("📝 Student Information")

input_data = {}

with st.form("student_prediction_form"):

    col1, col2 = st.columns(2)

    for index, column in enumerate(X.columns):

        current_col = col1 if index % 2 == 0 else col2

        with current_col:

            label = column.replace("_", " ").title()

            if column in numeric_features:

                minimum = int(X[column].min())
                maximum = int(X[column].max())
                default = int(X[column].median())

                input_data[column] = st.number_input(
                    label,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=1
                )

            else:

                options = sorted(
                    X[column].dropna().unique().tolist()
                )

                input_data[column] = st.selectbox(
                    label,
                    options=options
                )

    submitted = st.form_submit_button(
        "🔍 Predict Student Risk",
        use_container_width=True
    )


if submitted:

    student_df = pd.DataFrame([input_data])

    try:
        prediction = int(model.predict(student_df)[0])

        st.divider()
        st.subheader("📊 Prediction Result")

        if prediction == 1:

            st.error("⚠️ Prediction: At Risk")

            st.warning(
                "The model has identified this student "
                "as potentially at academic risk. "
                "Additional academic support may be helpful."
            )

        else:

            st.success("✅ Prediction: Not At Risk")

            st.info(
                "The model has not identified this student "
                "as being at academic risk."
            )

        st.caption(
            "This is an experimental machine learning prediction, "
            "not a final judgment about a student's academic future."
        )

    except Exception as error:

        st.error(f"Prediction Error: {error}")


st.divider()

st.caption(
    "Machine Learning Mini Project | "
    "Student Academic Risk Prediction System"
)

st.caption(
    "Dataset: UCI Student Performance Dataset (Mathematics)"
)