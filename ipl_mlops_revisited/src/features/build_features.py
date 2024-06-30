import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml

drop_row_value=yaml.safe_load(open('params.yaml','r'))['build_features']['drop_rows']
rolling_window=yaml.safe_load(open('params.yaml','r'))['build_features']['rolling_window']

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

def save_split_df(df,path):
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

def bowling_team(row):
    match_teams=row['match_name'].split('v')
    if row['current_innings']==match_teams[0]:
        return match_teams[1]
    else:
        return match_teams[0]


def feature_engineering(df):

    return(
        df.assign
        (total_score=
                    df.groupby(['match_id','innings_id'])['runs'].transform('sum'),
         batsman_total_runs=
                    df.groupby(['match_id','batsman1_name'])['wkt_batsman_runs'].transform('sum'),
         batsman_total_balls=
                    df.groupby(['match_id','batsman1_name'])['wkt_batsman_balls'].transform('sum'),
         current_runs=
                    df.groupby(['match_id','innings_id'])['runs'].transform('cumsum'),
         rolling_back_30balls_runs=
                    df.groupby(['match_id','innings_id'])['runs'].rolling(window=rolling_window,min_periods=1).sum().reset_index(level=[0,1],drop=True),
         rolling_back_30balls_wkts=
                    df.groupby(['match_id','innings_id'])['wicket_id'].rolling(window=rolling_window,min_periods=1).count().reset_index(level=[0,1],drop=True),
         bowling_team=df.apply(bowling_team,axis=1)
         ).rename(columns={'current_innings':'batting_team'})
         [['total_score','batting_team','bowling_team','over', 'ball','current_runs','rolling_back_30balls_runs','rolling_back_30balls_wkts']]
         

        
    )
 


df = load_data("data/raw/all_season_details.csv")
if df is not None:
    print(df.columns)
df_clean=clean_data(df)
df_final=feature_engineering(df_clean).iloc[drop_row_value:,:]
save_split_df(df_final,"data/processed/df_final.csv")

