import pandas as pd

df = pd.read_csv('training_data.csv')
df_sorted = df.sort_values(by='label')
df_sorted.to_csv('training_data_sorted.csv', index=False)