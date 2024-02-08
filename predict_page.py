import streamlit as st
import pickle
import numpy as np


def load_ml_model():
    with open('saved_steps.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


data = load_ml_model()

forest_model_loaded = data["forest_model"]
le_country = data["le_country"]
le_education = data["le_education"]
le_worktype = data["le_worktype"]
le_industry = data["le_industry"]
le_devtype = data["le_devtype"]


def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""We need some information to predict the salary. Please select options
    from the drop-down menus.""")

    countries = (
        "Select an Option",
        "United States of America",
        "Germany",
        "United Kingdom of Great Britain and Northern Ireland",
        "India",
        "Canada",
        "France",
        "Brazil",
        "Poland",
        "Netherlands",
        "Australia",
        "Spain",
        "Italy"
    )

    education = (
        "Select an Option",
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    type_of_work = (
        "Select an Option",
        "hybrid",
        "remote",
        "in-person",
    )

    industry_type = (
        "Select an Option",
        "technology",
        "financial services",
        "manufacturing and supply chain",
        "healthcare",
        "retail and consumer services"
    )

    dev_type = (
        "Select an Option",
        "Developer, Full-Stack",
        "Developer, Back-End",
        "Developer, Front-End",
        "Developer, Desktop / Enterprise",
        "Developer, Mobile",
        "Engineering Manager",
        "Developer, Embedded",
        "DevOps",
        "Data Engineer",
        "Data Scientist / ML",
        "Cloud Engineer"
    )

    country = st.selectbox("Country:", countries)
    education = st.selectbox("Education Level:", education)
    work_type = st.selectbox("Work Type:", type_of_work)
    industry_type = st.selectbox("Industry:", industry_type)
    dev_type = st.selectbox("Dev Type:", dev_type)
    experience = st.slider("Years of Experience:", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, work_type, industry_type, dev_type, experience]])
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_education.transform(X[:, 1])
        X[:, 2] = le_worktype.transform(X[:, 2])
        X[:, 3] = le_industry.transform(X[:, 3])
        X[:, 4] = le_devtype.transform(X[:, 4])
        X = X.astype(float)

        salary = forest_model_loaded.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
