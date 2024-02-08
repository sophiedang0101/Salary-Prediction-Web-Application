import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# Function to clean and shorten list of countries.
def shorten_country_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


def shorten_dev_types(types, cutoff):
    dev_type_map = {}
    for i in range(len(types)):
        if types.values[i] >= cutoff:
            dev_type_map[types.index[i]] = types.index[i]
        else:
            dev_type_map[types.index[i]] = 'Other (please specify)'
    return dev_type_map


# Function to clean and shorten list of industries.
def shorten_industry_types(types, cutoff):
    industry_types_map = {}
    for i in range(len(types)):
        if types.values[i] >= cutoff:
            industry_types_map[types.index[i]] = types.index[i]
    return industry_types_map


def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)


def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'


@st.cache_data
def load_data():
    survey_df_2022 = pd.read_csv("so_survey_2022.csv")
    survey_df_2023 = pd.read_csv("so_survey_2023.csv")
    survey_df_2020 = pd.read_csv("so_survey_2020.csv")
    survey_df_2021 = pd.read_csv("so_survey_2021.csv")
    merged_df = pd.concat([survey_df_2020, survey_df_2021, survey_df_2022, survey_df_2023], ignore_index=True)

    replace_dict = {
        "Hybrid (some remote, some in-person)": "hybrid",
        "Fully remote": "remote",
        "Full in-person": "in-person",
        "In-Person": "in-person"
    }

    # Replace specific values as per the mapping dictionary
    merged_df["RemoteWork"] = merged_df["RemoteWork"].replace(replace_dict)
    # Convert all values in the column to lowercase
    merged_df["RemoteWork"] = merged_df["RemoteWork"].str.lower()

    # Drop all other columns except specified ones.
    # Rename ConvertedCompYearly and RemoteWork columns.
    merged_df = merged_df[
        ["Country", "EdLevel", "YearsCodePro", "Employment", "RemoteWork", "ConvertedCompYearly", "Industry",
         "DevType"]]
    merged_df = merged_df.rename({"ConvertedCompYearly": "YearlySalary"}, axis=1)
    merged_df = merged_df.rename({"RemoteWork": "WorkType"}, axis=1)
    merged_df.head()

    # Drop all null and na values.
    merged_df = merged_df[merged_df["YearlySalary"].notnull()]
    merged_df = merged_df.dropna()
    # merged_df.isnull().sum()

    merged_df["Employment"] = merged_df["Employment"].replace("Employed, full-time", "full-time")
    merged_df = merged_df.drop("Employment", axis=1)

    country_map = shorten_country_categories(merged_df.Country.value_counts(), 600)
    merged_df['Country'] = merged_df['Country'].map(country_map)
    merged_df = merged_df[merged_df['Country'] != 'Other']

    merged_df = merged_df[merged_df["YearlySalary"] <= 250000]
    merged_df = merged_df[merged_df["YearlySalary"] >= 10000]

    dev_type_map = shorten_dev_types(merged_df.DevType.value_counts(), 300)
    merged_df['DevType'] = merged_df['DevType'].map(dev_type_map)
    merged_df = merged_df[~merged_df['DevType'].str.contains('Other', case=False, na=False)]

    dev_type_mapping = {
        'Developer, full-stack': 'Developer, Full-Stack',
        'Developer, back-end': 'Developer, Back-End',
        'Developer, front-end': 'Developer, Front-End',
        'Developer, desktop or enterprise applications': 'Developer, Desktop/Enterprise',
        'Developer, mobile': 'Developer, Mobile',
        'Engineering manager': 'Engineering Manager',
        'Developer, embedded applications or devices': 'Developer, Embedded',
        'DevOps specialist': 'DevOps',
        'Engineer, data': 'Data Engineer',
        'Data scientist or machine learning specialist': 'Data Scientist/ML',
        'Cloud infrastructure engineer': 'Cloud Engineer'
    }

    merged_df['DevType'] = merged_df['DevType'].map(dev_type_mapping)

    replace_dict = {
        'Information Services, IT, Software Development, or other Technology': 'technology',
        'Manufacturing, Transportation, or Supply Chain': 'manufacturing and supply chain',
        'Oil & Gas': 'oil and gas'
    }

    # # Replace specific values as per the mapping dictionary
    merged_df["Industry"] = merged_df["Industry"].replace(replace_dict)

    # Convert all values in the column to lowercase
    merged_df["Industry"] = merged_df["Industry"].str.lower()

    industry_map = shorten_industry_types(merged_df.Industry.value_counts(), 500)
    merged_df['Industry'] = merged_df['Industry'].map(industry_map)
    merged_df = merged_df[merged_df['Industry'] != 'other']

    merged_df["YearsCodePro"] = merged_df["YearsCodePro"].apply(clean_experience)
    merged_df["EdLevel"] = merged_df["EdLevel"].apply(clean_education)

    return merged_df


merged_df = load_data()


def show_explore_page():
    st.title("Explore Software Engineer Salaries")

    st.write(
        """
    ### Data from Stack Overflow Developer Surveys 2020-2023
    """
    )

    data = merged_df["Country"].value_counts()

    # Increase the size of the figure (adjust the values as needed)
    fig1, ax1 = plt.subplots(figsize=(10, 8))

    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=False, startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.write("""#### Number of Data from Different Countries""")

    st.pyplot(fig1)

    st.write(
        """
    #### Mean Salary Based On Country
    """
    )

    data = merged_df.groupby(["Country"])["YearlySalary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write(
        """
    #### Mean Salary Based On Experience
    """
    )

    data = merged_df.groupby(["YearsCodePro"])["YearlySalary"].mean().sort_values(ascending=True)
    st.line_chart(data)

    st.write(
        """
    ### Mean Salary Based On Education
        """
    )

    data = merged_df.groupby(["EdLevel"])["YearlySalary"].mean().sort_values(ascending=True)
    st.bar_chart(data)
