import pandas as pd
import os

def load_data(path):
    # Check if the file exists
    if not os.path.exists(path):
        print(f"Error: The file at path {path} does not exist.")
        return None

    # Try to load the CSV file
    try:
        df = pd.read_csv(path)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def save_cleaned_df(df,path):
    try:
        df.to_csv(path,index=False)
        print("File saved succesfully")
    except Exception as e:
        print(f"Error saving DataFrame: {e}")

def clean_data(df):
        return(
                 df.drop("comment_id",axis=1)
                 .assign(
                        home_team=lambda df_:(
                                df_
                                .home_team
                                .str.replace("PWI","PBKS")
                                .str.replace('GL','GT')
                                .str.replace("KXIP","PBKS")  
                                ),
                        away_team=lambda df_:(
                                df_
                                .away_team
                                .str.replace("PWI","PBKS")
                                .str.replace('GL','GT')
                                .str.replace("KXIP","PBKS")  
                                ),
                        current_innings=lambda df_:(
                                df_
                                .current_innings
                                .str.replace("PWI","PBKS")
                                .str.replace('GL','GT')
                                .str.replace("KXIP","PBKS")  
                                )               
                        )
              )

def feature_engineering(df):
    return df


df = load_data("data/raw/all_season_details.csv")
if df is not None:
    print(df.columns)
df_clean=clean_data(df)
df_fet_eng=feature_engineering(df_clean)
save_cleaned_df(df_fet_eng,"data/raw/cleaned.csv")