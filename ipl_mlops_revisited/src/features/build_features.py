import pandas as pd
import os
from sklearn.model_selection import train_test_split

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
         cumulative_runs=
                    df.groupby(['match_id','innings_id'])['runs'].transform('cumsum'),
         rolling_back_30balls_runs=
                    df.groupby(['match_id','innings_id'])['runs'].rolling(window=30,min_periods=1).sum().reset_index(level=[0,1],drop=True),
         rolling_back_30balls_wkts=
                    df.groupby(['match_id','innings_id'])['wicket_id'].rolling(window=30,min_periods=1).count().reset_index(level=[0,1],drop=True),
         bowling_team=df.apply(bowling_team,axis=1)
         ).rename(columns={'current_innings':'batting_team'})
         [['total_score','batting_team','bowling_team','rolling_back_30balls_runs','rolling_back_30balls_wkts']]
         

        
    )
 


df = load_data("data/raw/all_season_details.csv")
if df is not None:
    print(df.columns)
df_clean=clean_data(df)
df_final=feature_engineering(df_clean).iloc[30:,:]
save_split_df(df_final,"data/processed/df_final.csv")

'''X=df_fet_eng.drop(['total_score'],axis=1)
y=df_fet_eng['total_score']

X_train,X_test,y_tain,y_test_data=train_test_split(X,y,test_size=.2,random_state=22)
save_split_df(X_train,"data/processed/X_train.csv")
save_split_df(X_test,"data/processed/X_test.csv")
save_split_df(y_tain,"data/processed/y_train.csv")
save_split_df(y_test_data,"data/processed/y_test.csv")'''