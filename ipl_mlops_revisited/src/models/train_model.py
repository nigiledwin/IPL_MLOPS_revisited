import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pickle
import shap
import os
import matplotlib.pyplot as plt
import yaml

#import params.yaml and define variables
test_size=yaml.safe_load(open('params.yaml','r'))['train_model']['test_size']
n_estimators=yaml.safe_load(open('params.yaml','r'))['train_model']['n_estimators']
max_depth=yaml.safe_load(open('params.yaml','r'))['train_model']['max_depth']
learning_rate=yaml.safe_load(open('params.yaml','r'))['train_model']['learning_rate']

def preprocessing(X_train, y_train, X_test, y_test):
    # Define the ColumnTransformer correctly
    trf1 = ColumnTransformer([
                            ('team_ohe', OneHotEncoder(handle_unknown='ignore', drop='first'), [0, 1])
                            ], remainder='passthrough')
    
    model = XGBRegressor(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)
    pipe_xgb = Pipeline([
            ('trf1', trf1),
            ('model', model)
                ])
    pipe_xgb.fit(X_train, y_train)
    y_pred = pipe_xgb.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE calculation

    '''# SHAP explainability
    # Convert the transformed data to a dense format
    X_train_transformed = pipe_xgb.named_steps['trf1'].transform(X_train).toarray()
    X_test_transformed = pipe_xgb.named_steps['trf1'].transform(X_test).toarray()

    explainer = shap.Explainer(pipe_xgb.named_steps['model'], X_train_transformed)
    shap_values = explainer(X_test_transformed)

    # Save SHAP values for further analysis
    np.save('models/shap_values.npy', shap_values.values)
    shap.summary_plot(shap_values, X_test_transformed, show=False)
    plt.savefig('models/shap_summary_plot.png')'''
    return pipe_xgb, rmse

df_final = pd.read_csv("./data/processed/df_final.csv")
X = df_final.drop(['total_score'], axis=1)
y = df_final['total_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=22)

pipe_xgb, rmse = preprocessing(X_train, y_train, X_test, y_test)
print(rmse)

# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)

# Save the model to the 'models' directory
pickle.dump(pipe_xgb, open('models/pipe_xgb.pkl', 'wb'))
