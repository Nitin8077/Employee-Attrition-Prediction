import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Set page configuration
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

# --- Load and Prepare Data ---
@st.cache_data
def load_data():
    """Loads the dataset and prepares it for the model."""
    try:
        df = pd.read_csv("IBMdataset.csv")
    except FileNotFoundError:
        st.error("Error: The 'IBMdataset.csv' file was not found. Please make sure it's in the same directory.")
        return None, None, None

    # Convert 'Attrition' to numerical for modeling
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    # Select relevant features based on the notebook analysis
    features = [
        'Age', 'BusinessTravel', 'Department', 'EducationField', 'Gender',
        'JobRole', 'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked',
        'OverTime', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole'
    ]

    # Handle categorical variables with one-hot encoding
    df_encoded = pd.get_dummies(df[features], drop_first=True)
    
    X = df_encoded
    y = df['Attrition']
    
    return X, y, df

X, y, original_df = load_data()

# Check if data loaded successfully
if X is not None and y is not None:
    # Train the model
    @st.cache_resource
    def train_model(X_data, y_data):
        """Trains and returns the Random Forest model."""
        # Using a small train-test split to train the model quickly for the app
        X_train, _, y_train, _ = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    model = train_model(X, y)

    # Get feature names for later use
    model_features = X.columns
    
    # --- App UI ---
    st.title("IBM Employee Attrition Predictor")
    st.markdown("This application predicts the likelihood of an employee leaving the company based on their profile. Use the form on the left to input employee details.")
    
    # --- Sidebar for User Input ---
    st.sidebar.header("Employee Profile Input")

    # Dropdowns for categorical features
    department_options = original_df['Department'].unique()
    job_role_options = original_df['JobRole'].unique()
    marital_status_options = original_df['MaritalStatus'].unique()
    business_travel_options = original_df['BusinessTravel'].unique()
    education_field_options = original_df['EducationField'].unique()
    gender_options = original_df['Gender'].unique()
    overtime_options = original_df['OverTime'].unique()

    age = st.sidebar.slider("Age", 18, 60, 30)
    monthly_income = st.sidebar.slider("Monthly Income ($)", 1000, 20000, 6000)
    total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 5)
    years_at_company = st.sidebar.slider("Years At Company", 0, 40, 2)
    years_in_current_role = st.sidebar.slider("Years in Current Role", 0, 18, 2)
    num_companies_worked = st.sidebar.slider("Number of Companies Worked", 0, 9, 1)

    department = st.sidebar.selectbox("Department", department_options)
    job_role = st.sidebar.selectbox("Job Role", job_role_options)
    marital_status = st.sidebar.selectbox("Marital Status", marital_status_options)
    business_travel = st.sidebar.selectbox("Business Travel", business_travel_options)
    education_field = st.sidebar.selectbox("Education Field", education_field_options)
    gender = st.sidebar.selectbox("Gender", gender_options)
    overtime = st.sidebar.selectbox("OverTime", overtime_options)
    
    # --- Prediction Logic ---
    if st.sidebar.button("Predict Attrition"):
        # Create a dictionary for the input features
        input_data = {
            'Age': [age],
            'MonthlyIncome': [monthly_income],
            'TotalWorkingYears': [total_working_years],
            'YearsAtCompany': [years_at_company],
            'YearsInCurrentRole': [years_in_current_role],
            'NumCompaniesWorked': [num_companies_worked],
            'BusinessTravel': [business_travel],
            'Department': [department],
            'EducationField': [education_field],
            'Gender': [gender],
            'JobRole': [job_role],
            'MaritalStatus': [marital_status],
            'OverTime': [overtime]
        }
        
        # Create a dataframe from the input
        input_df = pd.DataFrame(input_data)
        
        # One-hot encode the input data, aligning with the model's features
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        
        # Ensure the columns are in the same order as the trained model's features
        # Add missing columns with a value of 0
        missing_cols = set(model_features) - set(input_encoded.columns)
        for c in missing_cols:
            input_encoded[c] = 0
            
        input_encoded = input_encoded[model_features]
        
        # Get the prediction and probability
        prediction = model.predict(input_encoded)
        prediction_proba = model.predict_proba(input_encoded)[0][1] * 100
        
        st.subheader("Prediction Result")
        
        # Display the result
        if prediction[0] == 1:
            st.error(f"Prediction: This employee is **LIKELY TO LEAVE**.")
            st.metric("Probability of Attrition", f"{prediction_proba:.2f}%")
        else:
            st.success("Prediction: This employee is **LIKELY TO STAY**.")
            st.metric("Probability of Attrition", f"{prediction_proba:.2f}%")
            
        # --- Explain the factors (Prescriptive Analysis) ---
        st.subheader("Key Factors Influencing this Prediction")
        st.markdown("The following factors from the employee's profile have a significant impact on this prediction, based on our model.")
        
        # Get feature importances from the model
        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': model_features, 'Importance': feature_importances})
        
        # Sort by importance and display top 5
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Display top 5 influential features
        for index, row in feature_importance_df.head(5).iterrows():
            feature_name = row['Feature'].replace('OverTime_Yes', 'OverTime').replace('_', ' ')
            st.write(f"- **{feature_name}**: (Importance: {row['Importance']:.4f})")

        st.markdown("---")
        st.info("Disclaimer: This model is for illustrative purposes and should not be used for making definitive business decisions without further validation.")