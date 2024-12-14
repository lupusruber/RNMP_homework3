import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
df = df.rename({"Diabetes_binary": "DiabetesBinary"}, axis=1)

df_offline, df_online = train_test_split(df, test_size=0.2, stratify=df.DiabetesBinary)

df_offline.to_csv("offline_data.csv", index=False)
df_online.drop(labels=["DiabetesBinary"], axis=1).to_csv("online_data.csv", index=False)
