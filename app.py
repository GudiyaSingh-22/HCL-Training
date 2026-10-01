import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI-Based Resume Screening System")
st.write(
    "Predict whether a candidate is likely to be shortlisted "
    "using resume-related features."
)

# Load dataset
df = pd.read_csv("ai_resume_screening.csv.csv")

# Features and target
X = df.drop("shortlisted", axis=1)
y = df["shortlisted"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            ["education_level"]
        )
    ],
    remainder="passthrough"
)

# Logistic Regression pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# Train model on complete dataset for the demo
model.fit(X, y)

st.divider()

st.subheader("Candidate Details")

col1, col2 = st.columns(2)

with col1:
    years_experience = st.number_input(
        "Years of Experience",
        min_value=0,
        max_value=100,
        value=5
    )

    skills_match_score = st.number_input(
        "Skills Match Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=0.1
    )

    education_level = st.selectbox(
        "Education Level",
        sorted(df["education_level"].unique())
    )

with col2:
    project_count = st.number_input(
        "Project Count",
        min_value=0,
        max_value=100,
        value=8
    )

    resume_length = st.number_input(
        "Resume Length",
        min_value=0,
        max_value=5000,
        value=550
    )

    github_activity = st.number_input(
        "GitHub Activity",
        min_value=0,
        max_value=10000,
        value=300
    )

candidate = pd.DataFrame([{
    "years_experience": years_experience,
    "skills_match_score": skills_match_score,
    "education_level": education_level,
    "project_count": project_count,
    "resume_length": resume_length,
    "github_activity": github_activity
}])

st.divider()

if st.button("🔍 Screen Candidate", type="primary"):

    prediction = model.predict(candidate)[0]

    probabilities = model.predict_proba(candidate)[0]
    classes = list(model.classes_)

    yes_probability = probabilities[classes.index("Yes")]

    if prediction == "Yes":
        st.success("✅ Candidate is likely to be shortlisted")
    else:
        st.warning("⚠️ Candidate is likely not to be shortlisted")

    st.metric(
        "Shortlisting Probability",
        f"{yes_probability * 100:.2f}%"
    )

st.divider()

st.subheader("Dataset Overview")

a, b, c = st.columns(3)

a.metric("Total Candidates", f"{len(df):,}")
b.metric("Features", "6")
c.metric("Model", "Logistic Regression")