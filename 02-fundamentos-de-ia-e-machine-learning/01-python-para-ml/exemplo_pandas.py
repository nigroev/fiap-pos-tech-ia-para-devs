import pandas as pd
data = pd.read_csv('data.csv')
mean_value = data['column_name'].mean()
print(mean_value)