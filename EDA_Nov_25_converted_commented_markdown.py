#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pprint import pprint as pprint
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.model_selection import GridSearchCV

#%%
import pandas as pd

# Sample data frame mimicking the user's data structure.
# In a real scenario, you would load the data from a CSV or Excel file.
data = pd.read_csv("data/Morningstar - European Mutual Funds.csv")

# Convert the data into a pandas DataFrame.
df = pd.DataFrame(data)



#%%
import pandas as pd

# Assuming 'df' is your pandas DataFrame
# Set pandas options to display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

#%%
len(df)

#%%
df.head()

#%%
df.describe()

#%%
pd.set_option('display.max_columns', None)
print(df.columns.to_list())

#%%
print(df.info())

#%% [markdown]
# # Question 1

#%% [markdown]
# What is the average ongoing cost of Mutual Funds with based on the rating?

#%%
average_costs_by_rating = df.groupby('rating')['ongoing_cost'].mean()

# Create a bar plot
average_costs_by_rating.plot(kind='bar', color='skyblue', figsize=(8, 6))

plt.xlabel('Rating')
plt.ylabel('Average Ongoing Cost')
plt.title('Average Ongoing Cost of Mutual Funds by Rating')
plt.xticks(rotation=0)  # Rotate the x-axis labels to be horizontal
plt.show()

#%% [markdown]
# As expected the high rating funds are more cost efficient

#%% [markdown]
# A well-managed fund with slightly higher fees might be a better choice than a poorly managed low-cost fund. The expertise and track record of the fund management team can be a crucial factor.

#%% [markdown]
# # Question 2

#%% [markdown]
# How many funds have achieved a sustainability score above 80 percentile and also have a top
# quartile performance rating?

#%%

# Define the 80th percentile for sustainability score
sustainability_80th_percentile = df['sustainability_score'].quantile(0.8)

# Filter the DataFrame for funds with a sustainability score above the 80th percentile
high_sustainability_funds = df[df['sustainability_score'] > sustainability_80th_percentile]

# Further filter for funds that also have a top quartile performance rating (assuming 1 is the top quartile)
top_funds = high_sustainability_funds[high_sustainability_funds['performance_rating'] == 5]

# Count the number of such funds
number_of_top_funds = top_funds.shape[0]

number_of_top_funds

#%%
hig_sus_funds_count = high_sustainability_funds.groupby('performance_rating')["ticker"].count()
total_funds = high_sustainability_funds.shape[0]
percent_by_rating = (hig_sus_funds_count / total_funds) * 100
# Create a bar plot
percent_by_rating.plot(kind='bar', color='skyblue', figsize=(8, 6))


plt.xlabel('Performance rating')
plt.ylabel('Sustainability_score')
plt.title('No of Mutual Funds with high sustanability score for each rating')
plt.xticks(rotation=0)  # Rotate the x-axis labels to be horizontal
plt.show()

#%%
percent_by_rating

#%% [markdown]
# ### Analysis:
# 
# * The largest bar is above "3.0", suggesting that the majority of funds with a moderate performance rating also have high sustainability scores.
# * The second-largest group is funds with a "4.0" rating, indicating that many above-average performance-rated funds also prioritize sustainability.
# 
# * Interestingly, the bar for the highest performance rating ("5.0") is the shortest, which might imply that fewer top-rated funds have high sustainability scores compared to those with moderate ratings.
# * There appears to be a trend where the moderate-performing funds (ratings "2.0" to "4.0") have more high sustainability scores than the lowest ("1.0") and highest ("5.0") rated funds.
# #### Implications: 
# This could indicate that the highest-performing funds may not necessarily prioritize sustainability to the same extent as moderately rated funds, or it could reflect the distribution of sustainability initiatives across different performance levels.
# ##### Considerations for Investors : 
# * For investors who prioritize sustainability, this graph suggests that they might find more options with high sustainability scores among funds with moderate performance ratings rather than the highest-rated funds.

#%% [markdown]
# # Question 3
# 

#%% [markdown]
# How do funds with a high environmental score (above 70) compare in terms of ROA,
# ROE, and ROIC to funds with a low environmental score (below 30)?

#%%
percentiles_roa = df['roa'].quantile([0.25, 0.5, 0.75])

percentiles_roa

#%%
percentiles_roe = df['roe'].quantile([0.25, 0.5, 0.75])

percentiles_roe

#%%
percentiles_roic = df['roic'].quantile([0.25, 0.5, 0.75])

percentiles_roic

#%%
percentiles_env_score = df['environmental_score'].quantile([0.25, 0.5, 0.75])

percentiles_env_score

#%%
print(percentiles_env_score.iloc[2])

#%%

# Define the high and low environmental score thresholds.
high_score_threshold = percentiles_env_score.iloc[2]
low_score_threshold = percentiles_env_score.iloc[0]

# Calculate the average ROA, ROE, and ROIC for high and low environmental score funds.
high_score_funds = df[df['environmental_score'] > high_score_threshold]
low_score_funds = df[df['environmental_score'] < low_score_threshold]

# Calculate averages
high_score_averages = high_score_funds[['roa', 'roe', 'roic']].mean()
low_score_averages = low_score_funds[['roa', 'roe', 'roic']].mean()

high_score_averages, low_score_averages


#%%
# Create a bar plot
fig, ax = plt.subplots()
bar_width = 0.35
opacity = 0.8

# Convert the index to a numerical array for plotting
index = np.arange(len(high_score_averages))

bar1 = plt.bar(index, high_score_averages, bar_width, alpha=opacity, color='b', label='High Environmental Score')
bar2 = plt.bar(index + bar_width, low_score_averages, bar_width, alpha=opacity, color='g', label='Low Environmental Score')

plt.xlabel('Metrics')
plt.ylabel('Averages')
plt.title('Average ROA, ROE, ROIC by Environmental Score')
plt.xticks(index + bar_width / 2, high_score_averages.index)
plt.legend()

plt.tight_layout()
plt.show()

#%% [markdown]
# * The funds with high environmental score has lower performance compared to the funds with low environmental score as expected

#%% [markdown]
# # comparing high and low environmental score funds using boxplots

#%%
# Combine high and low score funds into a single DataFrame for plotting
high_score_funds['Score Group'] = 'High Score'
low_score_funds['Score Group'] = 'Low Score'
combined_data = pd.concat([high_score_funds, low_score_funds])

# Melting the DataFrame to have a long format suitable for seaborn boxplot
melted_data = combined_data.melt(id_vars=[ 'Score Group'], value_vars=['roa', 'roe', 'roic'], 
                                 var_name='Metric', value_name='Value')

# Creating the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Metric', y='Value', hue='Score Group', data=melted_data)
plt.title('Boxplot of ROA, ROE, and ROIC for High and Low Environmental Score Funds')
plt.show()

#%%
melted_data

#%% [markdown]
# We observe the same thing.
# * The funds with high environmental score has lower performance compared to the funds with low environmental score as expected

#%% [markdown]
# # Question 4

#%% [markdown]
# What is the average fund size for each fund category?

#%%
# Group by 'fund_category' and calculate the average 'fund_size' for each group
average_fund_size_by_category = df.groupby('category')['fund_size'].mean()

average_fund_size_by_category

#%%
percent_nan = df['category'].isna().mean() * 100

#%%
percent_nan

#%%
percent_nan = df['fund_size'].isna().mean() * 100

#%%
percent_nan

#%%
df = df.dropna(subset=['fund_size'])

#%%
percent_nan = df['fund_size'].isna().mean() * 100
percent_nan

#%%
# Find the top 10 fund categories based on average fund size
top_10_categories = average_fund_size_by_category.nlargest(10)

# Find the bottom 10 fund categories based on average fund size
bottom_10_categories = average_fund_size_by_category.nsmallest(10)

top_10_categories, bottom_10_categories

#%%
# Create a bar plot
top_10_categories.plot(kind='bar', color='skyblue', figsize=(8, 6))


plt.xlabel('Categories')
plt.ylabel('Fund size')
plt.title('Funds size for top 10 category')
plt.xticks(rotation=90)  # Rotate the x-axis labels to be horizontal
plt.show()

#%%
# Create a bar plot
bottom_10_categories.plot(kind='bar', color='skyblue', figsize=(8, 6))


plt.xlabel('Categories')
plt.ylabel('Fund size')
plt.title('Funds size for bottom 10 category')
plt.xticks(rotation=90)  # Rotate the x-axis labels to be horizontal
plt.show()

#%% [markdown]
# Most of the largest funds are japanese fund and most of the bottom funds are from middle East.

#%% [markdown]
# # Question 5

#%% [markdown]
# How has the average Price to Earnings (P/E) ratio changed over the past five years
# across different equity styles?

#%%
percent_nan = df[['equity_style','nav_per_share','fund_return_2015','fund_return_2016','fund_return_2017','fund_return_2018','fund_return_2019']].isna().mean() * 100
percent_nan

#%%
len(df)

#%%
df = df.dropna(subset=['equity_style','nav_per_share','fund_return_2015','fund_return_2016','fund_return_2017','fund_return_2018','fund_return_2019'])

#%%
len(df)

#%%
#df = pd.DataFrame(df)

# Assuming the return can be used as a proxy for earnings growth (very speculative)
df['eps_2015'] = df['nav_per_share'] * df['fund_return_2015']
df['eps_2016'] = df['nav_per_share'] * df['fund_return_2016']
df['eps_2017'] = df['nav_per_share'] * df['fund_return_2017']
df['eps_2018'] = df['nav_per_share'] * df['fund_return_2018']
df['eps_2019'] = df['nav_per_share'] * df['fund_return_2019']
# Calculate for other years similarly...

# Now calculate P/E-like ratio for each year
df['pe_2015'] = df['nav_per_share'] / df['eps_2015']
df['pe_2016'] = df['nav_per_share'] / df['eps_2016']
df['pe_2017'] = df['nav_per_share'] / df['eps_2017']
df['pe_2018'] = df['nav_per_share'] / df['eps_2018']
df['pe_2019'] = df['nav_per_share'] / df['eps_2019']

# Calculate for other years similarly...

# Aggregate by equity style
pe_by_style_and_year = df.groupby('equity_style').agg({'pe_2015': 'mean', 'pe_2016': 'mean',  'pe_2017': 'mean', 'pe_2018': 'mean', 'pe_2019': 'mean'})
# Add other years similarly...

#%%
df[['eps_2015','eps_2016','pe_2015','pe_2016', 'pe_2017', 'pe_2018','pe_2019']].isna().mean() * 100

#%%
pe_by_style_and_year

#%%
for col in ['pe_2015', 'pe_2016', 'pe_2017', 'pe_2018','pe_2019']:
    df[col].replace([np.inf, -np.inf], np.nan, inplace=True)


#%%
df[['eps_2015','eps_2016','pe_2015','pe_2016', 'pe_2017', 'pe_2018','pe_2019']].isna().mean() * 100

#%%
df = df.dropna(subset=['eps_2015','eps_2016','pe_2015','pe_2016', 'pe_2017', 'pe_2018','pe_2019'])

#%%
df[['eps_2015','eps_2016','pe_2015','pe_2016', 'pe_2017', 'pe_2018','pe_2019']].isna().mean() * 100

#%%
len(df)

#%%
pe_by_style_and_year = df.groupby('equity_style').agg({'pe_2015': 'mean', 'pe_2016': 'mean',  'pe_2017': 'mean', 'pe_2018': 'mean', 'pe_2019': 'mean'})
pe_by_style_and_year

#%%
# Convert the DataFrame from wide format to long format
pe_melted = pe_by_style_and_year.reset_index().melt(id_vars='equity_style', value_vars = ['pe_2015','pe_2016', 'pe_2017', 'pe_2018','pe_2019'], var_name='Year', value_name='PE Ratio')

# Rename the columns appropriately
#pe_melted.rename(columns={'index': 'Equity Style'}, inplace=True)

# Create a seaborn bar plot
plt.figure(figsize=(8, 6))
sns.barplot(data=pe_melted, x='equity_style', y='PE Ratio', hue='Year')
plt.title('Average PE Ratio by Equity Style')

#%%
pe_melted

#%% [markdown]
# 
# 
# ### Graph 1: Average PE Ratio by Equity Style
# 
# #### Variation Across Years and Styles:
# 
# There is variation in the average P/E ratio both across different equity styles and across the years from 2015 to 2019.
# The P/E ratios for 'Growth' and 'Value' equity styles have fluctuated over the years, whereas 'Blend' seems to have a more stable P/E ratio over time.
# #### Negative Values:
# 
# There are negative P/E ratios displayed in the graph. Negative P/E ratios can occur if the earnings (denominator in the P/E calculation) are negative, meaning the company or fund reported a loss.
# Trends:
# 
# For 'Blend' equity style, there was a significant dip in 2018, which suggests a decrease in price relative to earnings or an increase in earnings relative to price.
# 'Growth' style shows a notable decrease in P/E ratio from 2018 to 2019, indicating it became less expensive or earnings growth outpaced the price increase.
# 'Value' style shows an increase in P/E ratio from 2018 to 2019, indicating it may have become more expensive relative to its earnings or that the earnings growth was slower than the price increase.
# 
# **We enquired and found out that these are the reasons for economic downturn in 2018**
# * The trumps's trade war with china
# * Federal reserve raising interest rate quickly
# * Inflated company earnings.
# 

#%%

# Create a seaborn bar plot
plt.figure(figsize=(8, 6))
sns.barplot(data=pe_melted, x='Year', y='PE Ratio', hue='equity_style')
plt.title('Average PE Ratio by Equity Style')

#%% [markdown]
# Growth' style has a clear pattern of decreasing P/E ratios over the years, which could imply a decrease in the valuation multiples or an increase in earnings growth.
# 'Blend' and 'Value' styles do not show a clear trend over the years, suggesting inconsistent changes in their P/E ratios.

#%% [markdown]
# 2018 Anomaly:
# 
# The year 2018 stands out with significantly negative P/E ratios for all equity styles. This could indicate a widespread market event that caused losses in earnings or an anomaly in the data.
# Stability in 'Value' Style:

#%% [markdown]
# # what is the funds return growth from 2015 to 2020 for different equity_size, can you code line graph

#%%
# Analyzing the funds return growth from 2015 to 2020 for different equity sizes

# Assuming the relevant columns for yearly returns are 'fund_return_2015', ..., 'fund_return_2020'
# and 'equity_size' indicates the equity size of the fund

# Check if the relevant columns exist
year_columns = ['fund_return_2015', 'fund_return_2016', 'fund_return_2017', 
                'fund_return_2018', 'fund_return_2019']

if 'equity_style' in df.columns and all(year in df.columns for year in year_columns):
    # Grouping by equity size and calculating the average return for each year
    avg_return_by_equity_size = df.groupby('equity_style')[year_columns].mean()

    # Plotting the line graph
    plt.figure(figsize=(12, 8))
    for equity_size in avg_return_by_equity_size.index:
        plt.plot(avg_return_by_equity_size.columns, avg_return_by_equity_size.loc[equity_size, :], 
                 marker='o', label=equity_size)

    plt.xlabel('Year')
    plt.ylabel('Average Fund Return (%)')
    plt.title('Fund Return Growth from 2015 to 2020 by Equity style')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    plt.text(0.5, 0.5, 'Some specified columns are missing from the DataFrame.', 
             horizontalalignment='center', verticalalignment='center')
    plt.title('Data Unavailable')
    plt.show()


#%% [markdown]
# 2018 has the least growth of all years.
# In 2019, growth style gives more growth as expected.

#%% [markdown]
# # Question 6 

#%% [markdown]
# What is the correlation between the sustainability score and fund trailing returns over 1,
# 3, and 5 years?

#%%
correlation_data = df[['sustainability_score', 'fund_trailing_return_ytd', 'fund_trailing_return_3years', 'fund_trailing_return_5years']].corr()

# The resulting DataFrame 'correlation_data' will contain the correlation coefficients between all pairs of these variables
correlation_data

#%%
# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_data, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#%% [markdown]
#  These variables 'fund_trailing_return_ytd', 'fund_trailing_return_3years', 'fund_trailing_return_5years' are highly correlated.

#%% [markdown]
# But the sustanibility score is not correlated with them

#%% [markdown]

 # Question 7
# Calculate the average fund return for each risk rating category during the last quarter

#%%
import pandas as pd

# Assuming df is your DataFrame
# Replace 'your_dataframe.csv' with the actual file name if you are reading from a CSV file
# df = pd.read_csv('your_dataframe.csv')

# Group by 'risk_rating' and calculate the mean of 'fund_return_2019'
average_returns = df.groupby('risk_rating')['fund_return_2019'].mean()

print(average_returns)


#%%
# Creating the bar graph
plt.figure(figsize=(10, 6))
plt.bar(average_returns.index, average_returns, color='skyblue')

plt.xlabel('Risk Rating')
plt.ylabel('Average Fund Return in 2019 (%)')
plt.title('Average Fund Return in 2019 by Risk Rating')
plt.xticks(list(average_returns.index))
plt.show()

#%% [markdown]
# As we can see the funds with high risk has high returns in the year 2019

#%% [markdown]
# # Question 8
# 
# what are the fund percentage in each of the different credit ratings like a,aa,aaa

#%%
df.head(50).to_excel('data/first_50_rows.xlsx', index=False)

#%%
average_credit_a = df['credit_a'].mean()
average_credit_aa = df['credit_aa'].mean()
average_credit_aaa = df['credit_aaa'].mean()

#%%
credit_ratings = ['A', 'AA', 'AAA']
average_percentages = [average_credit_a, average_credit_aa, average_credit_aaa]

# Creating the bar graph
plt.figure(figsize=(10, 6))
plt.bar(credit_ratings, average_percentages, color='teal')

plt.xlabel('Credit Rating')
plt.ylabel('Average Percentage of Assets (%)')
plt.title('Average Percentage of Fund Assets in Different Credit Ratings')
plt.xticks(credit_ratings)
plt.show()

#%% [markdown]
# Most of the funds are in AAA ratings. People prefer their assets to be highly rated funds.

#%% [markdown]
# # Question 10

#%% [markdown]
#  For funds with a high risk rating, what investment strategies are commonly employed?

#%%
# Filter the DataFrame for high risk rating funds
high_risk_funds = df[df['risk_rating'] == df['risk_rating'].max()]

# Extracting the investment strategies
investment_strategies_high_risk = high_risk_funds['investment_strategy'].value_counts()

pprint(investment_strategies_high_risk)

#%% [markdown]
# ### Strategies followed by the high risk rating Funds.
# 
# US Growth Funds Investment Objective: This strategy seeks long-term capital appreciation, primarily through investing in securities issued by US companies and, to a lesser extent, in securities from non-US companies. The criteria for considering a company to be from a particular country or region includes its principal securities trading market location, revenue sources, or country of organization.
# 
# European Property Funds Investment Objective: This strategy aims for long-term capital appreciation by investing in the equity securities of companies in the European real estate industry. This includes property development companies, those engaged in ownership of income-producing property, and collective investment vehicles with property exposure.
# 
# Investing in Asian (Excluding Japan) Equities: One strategy focuses on achieving capital appreciation by investing principally in the equity securities of companies domiciled in Asia (excluding Japan) or with significant Asian operations. This may include a mix of equities, fixed income securities, and other instruments.
# 
# Investing in Asian (Excluding Japanese) Equities: Another strategy involves long-term capital growth through investment in a portfolio of Asian (excluding Japanese) equities. This includes companies domiciled in, based in, or conducting a major part of their business in Asia, including both developed markets and emerging markets.
# 
# These investment strategies represent a focus on geographic diversification and specific industry sectors, reflecting the varied approaches taken by high risk-rated funds to achieve capital appreciation. The count next to each strategy indicates the number of funds employing that particular strategy.


#
#%% [markdown]
# # Question 11

#%% [markdown]
# Which fund category has shown the most improvement in average return over the past three years?


#%%
# Calculating the improvement in average return for each fund category over the past three years

# Assuming the relevant columns for yearly returns are 'fund_return_2019', 'fund_return_2018', and 'fund_return_2017'
# Checking if these columns exist in the DataFrame
if all(item in df.columns for item in ['fund_return_2019', 'fund_return_2018', 'fund_return_2017']):
    # Calculating average returns for each year and each category
    avg_return_2019 = df.groupby('category')['fund_return_2019'].mean()
    avg_return_2018 = df.groupby('category')['fund_return_2018'].mean()
    avg_return_2017 = df.groupby('category')['fund_return_2017'].mean()

    # Calculating improvement from 2017 to 2019
    improvement = avg_return_2019 - avg_return_2017

    # Identifying the category with the most improvement
    most_improved_category = improvement.idxmax()
    most_improvement_value = improvement.max()
else:
    most_improved_category = None
    most_improvement_value = None

most_improved_category, most_improvement_value


#%% [markdown]
# Russian Equity has shown the most improvement in average return over the past
# three years, 39% from 2017 to 2019

#%%
# Creating box plots to compare 'fund_return_2019' across different categorical variables

# Selecting a few categorical columns for the analysis
categorical_columns = ['category', 'fund_benchmark', 'morningstar_benchmark']


#%%
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pretty_categorical_analysis(df, column, top_n=10):
    """Plot aesthetically improved box plots for the top N categories in a given column."""
    top_categories = df[column].value_counts().head(top_n).index
    filtered_df = df[df[column].isin(top_categories)]
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x=column, y='fund_return_2019', data=filtered_df, palette='viridis')
    plt.title(f'Fund Return 2019 by {column}', fontsize=16, fontweight='bold')
    plt.xlabel(column, fontsize=14)
    plt.ylabel('Fund Return 2019 (%)', fontsize=14)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# List of categorical columns to plot
categorical_columns = ['category', 'fund_benchmark', 'morningstar_benchmark']

# Plotting for each categorical column with enhanced aesthetics
for col in categorical_columns:
    plot_pretty_categorical_analysis(df, col)


#%% [markdown]
# ### Analysis from category
# 
# * Global Large cap Growth equity performs better.
# * GBP Moderate allocation has lowest fund return
# * Japan large cap equity performs bad and indicates it has not returned back from the fall from late 1990s
# 


