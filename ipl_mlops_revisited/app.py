import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Load the trained pipeline
pipe = pickle.load(open('models/pipe.pkl', 'rb'))
df = pd.read_csv("./data/processed/df_final.csv")

# Function to preprocess input data
def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])
    return df

def create_shap_explainer(pipe, data):
    transformer = pipe.named_steps['trf1']
    model = pipe.named_steps['model']
    
    # Transform the input data
    X_transformed = transformer.transform(data)
    
    # Convert sparse matrix to dense array if necessary
    if isinstance(X_transformed, np.ndarray):
        X_transformed_dense = X_transformed
    else:
        X_transformed_dense = X_transformed.toarray()  # Convert sparse matrix to dense array
    
    # Get feature names
    feature_names = transformer.get_feature_names_out()
    
    # Create SHAP explainer
    explainer = shap.Explainer(model, X_transformed_dense, feature_names=feature_names)
    
    return explainer
# Main Streamlit app
def main():
    st.title('Cricket Prediction App')
    st.markdown('Select teams and enter values for prediction')

    # Initialize session state
    if 'user_input' not in st.session_state:
        st.session_state.user_input = None

    # Sidebar navigation
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Prediction', 'SHAP Explanation'])

    if page == 'Prediction':
        render_prediction_page()
    elif page == 'SHAP Explanation':
        render_shap_explanation()

# Function to render prediction page
def render_prediction_page():
    st.subheader('Prediction Page')
    # Dropdowns to select teams
    batting_team = st.selectbox('Select Batting Team', options=df['batting_team'].unique())
    bowling_team = st.selectbox('Select Bowling Team', options=df['bowling_team'].unique())

    # Input fields for other features
    over = st.number_input('Over', min_value=1, step=1)
    ball = st.number_input('Ball', min_value=1, step=1)
    current_runs = st.number_input('Current Runs', min_value=0, step=1)
    rolling_back_30balls_runs = st.number_input('Rolling Back 30 Balls Runs', min_value=0, step=1)
    rolling_back_30balls_wkts = st.number_input('Rolling Back 30 Balls Wickets', min_value=0, step=1)

    # Create a dictionary with user inputs
    user_input = {
        'batting_team': batting_team,
        'bowling_team': bowling_team,
        'over': over,
        'ball': ball,
        'current_runs': current_runs,
        'rolling_back_30balls_runs': rolling_back_30balls_runs,
        'rolling_back_30balls_wkts': rolling_back_30balls_wkts
    }

    # Predict button
    if st.button('Predict'):
        try:
            # Preprocess the input data
            input_data = preprocess_input(user_input)
            
            # Store input data in session state
            st.session_state.user_input = input_data
            
            # Display input data
            st.write('Input Data:')
            st.write(input_data)

            # Transform input data using the pipeline
            X_transformed = pipe.named_steps['trf1'].transform(input_data)
            
            # Ensure X_transformed has the correct shape and features
            st.write('Transformed Data Shape:')
            st.write(X_transformed.shape)  # Check shape of transformed data
            st.write('Transformed Data Columns:')
            st.write(pipe.named_steps['trf1'].get_feature_names_out())  # Get transformed column names
            
            # Perform prediction
            prediction = pipe.predict(input_data)
            st.write('Prediction:', prediction)
            
            # If Predict button is clicked and prediction is successful, show link to SHAP explanation page
            if st.button('View SHAP Explanation'):
                st.session_state.show_shap_explanation = True

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Function to render SHAP explanation page
def render_shap_explanation():
    st.subheader('SHAP Explanation Page')
    try:
        # Retrieve stored input data from session state
        input_data = st.session_state.user_input
                
        if input_data is None:
            st.warning('No prediction data available. Please make a prediction first.')
            return

        # Ensure data types are appropriate and handle NaN values if necessary
         # Convert all columns to float (adjust as per your data types)

        # Transform input data using the pipeline
        
        input_transfered = pipe.named_steps['trf1'].transform(input_data)
        st.write(input_data)
        st.write(input_transfered)
        # Convert sparse matrix to dense array if necessary
        if isinstance(input_transfered, np.ndarray):
            input_dense = input_transfered
        else:
            input_dense = input_transfered.toarray()
        

        
        # Create SHAP explainer
        explainer = create_shap_explainer(pipe, df.drop(columns='total_score'))  # Pass input_data directly if it's already transformed correctly

        # Display SHAP values
        st.write(explainer.model)
        #st.write('SHAP Values:')
        shap_values = explainer(input_dense)
        st.write(shap_values)
        shap.waterfall_plot(shap.Explanation(values=shap_values[0].values, base_values=shap_values[0].base_values, data=input_dense[0],feature_names=pipe.named_steps['trf1'].get_feature_names_out()),show=False)
        st.pyplot()
    except Exception as e:
        st.error(f"Error during SHAP explanation: {e}")


if __name__ == '__main__':
    main()
