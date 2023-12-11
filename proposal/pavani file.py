#%%
import pandas as pd

# Assuming 'df' is your DataFrame
df = pd.read_csv('Morningstar - European Mutual Funds.csv')

# To get the number of rows and columns
num_rows, num_columns = df.shape

print("Number of Rows:", num_rows)
print("Number of Columns:", num_columns)
# %%
import pandas as pd

# Assuming df is your DataFrame
# Replace 'df' with the actual variable name of your DataFrame
column_names = df.columns

# Print all column names
print("Column Names:")
for column in column_names:
    print(column)

# %%
#7. Calculate the average fund return for each risk rating category during the last quarter.

average_returns_by_category = df.groupby('category')['fund_return_2019_q4'].mean()
average_returns_by_category
# %%
