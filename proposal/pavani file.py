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
#8. What percentage of funds have a credit quality of 'AAA'?


# Assuming df is your DataFrame
# Replace 'credit_aaa' and 'fund_name' with the actual column names
credit_aaa_column = 'credit_aaa'
fund_name_column = 'fund_size'

# Filter funds with a credit quality of 'AAA'
aaa_funds = df[df[credit_aaa_column] == 'AAA']

# Calculate the percentage of funds with 'AAA' credit quality
percentage_aaa_funds = (len(aaa_funds) / len(df)) * 100

# Display the result
print(f"Percentage of funds with 'AAA' credit quality: {percentage_aaa_funds:.2f}%")
# %%
#9. Identify the top 10 funds in terms of 5-year trailing return within each risk rating category.
#Among funds with a "Gold" analyst rating, which fund categories are most prevalent?

# Assuming df is your DataFrame
# Replace 'fund_trailing_return_5years', 'risk_rating', 'analyst_rating', and 'category' with actual column names
return_5years_column = 'fund_trailing_return_5years'
risk_rating_column = 'risk_rating'
analyst_rating_column = 'analyst_rating'
category_column = 'category'

# Filter funds with a "Gold" analyst rating
gold_rated_funds = df[df[analyst_rating_column] == 'Gold']

# Identify the top 10 funds in each risk rating category based on 5-year trailing return
top10_funds_by_risk = df.groupby(risk_rating_column).apply(lambda x: x.nlargest(10, return_5years_column))

# Determine the most prevalent fund categories among "Gold" rated funds
most_prevalent_categories = gold_rated_funds[category_column].value_counts().head(5)

# Display the top 10 funds in each risk rating category
print("Top 10 funds in terms of 5-year trailing return within each risk rating category:")
print(top10_funds_by_risk)

# Display the most prevalent fund categories among "Gold" rated funds
print("\nMost prevalent fund categories among 'Gold' rated funds:")
print(most_prevalent_categories)
# %%
