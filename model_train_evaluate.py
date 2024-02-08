import streamlit as st
import pandas as pd
from explore_page import clean_experience, clean_education, shorten_country_categories, shorten_dev_types, \
    shorten_industry_types
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV


@st.cache_data
def load_data():
    survey_df_2020 = pd.read_csv("so_survey_2020.csv")
    survey_df_2021 = pd.read_csv("so_survey_2021.csv")
    survey_df_2022 = pd.read_csv("so_survey_2022.csv")
    survey_df_2023 = pd.read_csv("so_survey_2023.csv")
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

    country_map = shorten_country_categories(merged_df.Country.value_counts(), 1000)
    merged_df["Country"] = merged_df["Country"].map(country_map)

    merged_df = merged_df[merged_df["YearlySalary"] <= 250000]
    merged_df = merged_df[merged_df["YearlySalary"] >= 10000]
    merged_df = merged_df[merged_df["Country"] != "Other"]

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

    le_education = LabelEncoder()
    merged_df['EdLevel'] = le_education.fit_transform(merged_df['EdLevel'])

    le_country = LabelEncoder()
    merged_df['Country'] = le_country.fit_transform(merged_df['Country'])

    le_worktype = LabelEncoder()
    merged_df['WorkType'] = le_worktype.fit_transform(merged_df['WorkType'])

    le_industry = LabelEncoder()
    merged_df['Industry'] = le_industry.fit_transform(merged_df['Industry'])

    le_devtype = LabelEncoder()
    merged_df['DevType'] = le_devtype.fit_transform(merged_df['DevType'])

    return merged_df


merged_df = load_data()


def show_model_train_page():
    st.title("Salary Prediction Random Forest Regressor Model")
    st.subheader("Training and Evaluation")
    if st.button("Train the Model"):
        # Train-test split
        train_df, test_df = train_test_split(merged_df, test_size=0.2)

        # Features and target variable
        X_train = train_df.drop("YearlySalary", axis=1)
        y_train = train_df["YearlySalary"]

        X_test = test_df.drop("YearlySalary", axis=1)
        y_test = test_df["YearlySalary"]

        # Random Forest Regressor
        forest_regressor = RandomForestRegressor()
        forest_regressor.fit(X_train, y_train)

        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [15, 20, 25],
            'min_samples_split': [8, 10, 12],
            'min_samples_leaf': [3, 4, 5],
            'max_features': ['auto', 'sqrt', 'log2']
        }

        grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                                   param_grid=param_grid,
                                   scoring='neg_mean_absolute_error',
                                   cv=5,
                                   n_jobs=-1)

        grid_search.fit(X_train, y_train)

        st.success("Model trained successfully!")

        st.subheader("Best Hyperparameters")
        st.write(f"{grid_search.best_params_}")

        # Get the best model from the grid search
        best_forest_model = grid_search.best_estimator_

        # Evaluate the best model on the test set
        best_model_predictions = best_forest_model.predict(X_test)

        # Calculate regression metrics
        mae = mean_absolute_error(y_test, best_model_predictions)
        mse = mean_squared_error(y_test, best_model_predictions)
        rmse = root_mean_squared_error(y_test, best_model_predictions)
        r2 = r2_score(y_test, best_model_predictions)

        st.write(f"Mean Absolute Error (MAE): {mae}")
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse}")
        st.write(f"R-squared: {r2}")

        # Feature importances
        st.subheader("Feature Importances")
        importances = best_forest_model.feature_importances_
        features = ['EdLevel', 'Country', 'WorkType', 'YearsCodePro', 'Industry', 'DevType']

        bar_width = 0.5

        # Plotting feature importances with adjusted width
        plt.bar(features, importances, width=bar_width)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        # plt.title('Feature Importances')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        st.pyplot(plt)

        st.subheader("Actual vs. Predicted Salaries")

        # Scatter plot
        actual_salaries = test_df['YearlySalary']
        predicted_salaries = best_forest_model.predict(X_test)

        # Create a scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(actual_salaries, predicted_salaries, alpha=0.5, label='Data Points')
        plt.plot([min(actual_salaries), max(actual_salaries)], [min(actual_salaries), max(actual_salaries)],
                 color='red', linestyle='--', linewidth=2, label='Diagonal Line (y = x)')
        # plt.title('Actual vs. Predicted Salaries')
        plt.xlabel('Actual Salaries')
        plt.ylabel('Predicted Salaries')
        plt.legend()
        plt.grid(True)
        plt.show()
        st.pyplot(plt)

        st.subheader("Distribution of Residuals")
        residuals = actual_salaries - predicted_salaries

        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, edgecolor='black')
        plt.title('Distribution of Residuals')
        plt.xlabel('Residuals (Actual - Predicted)')
        plt.ylabel('Frequency')
        plt.grid(True)
        st.pyplot(plt)
