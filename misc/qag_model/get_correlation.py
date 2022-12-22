import pandas as pd

df_qae = pd.read_csv("summary/summary.qae.average.csv")
df_qag = pd.read_csv("summary/summary.qag.csv")
df_qag.pop('Model')
df_qag.pop('Data')
df = pd.merge(left=df_qae, right=df_qag, on=['Language Model', 'Type'])
corr = df.corr()
corr.to_csv("summary/correlation.csv")
