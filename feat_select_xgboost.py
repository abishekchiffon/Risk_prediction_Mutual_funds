

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import pointbiserialr
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import kruskal

import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.linear_model import ElasticNet



from sklearn.model_selection import GridSearchCV



#%% [markdown]
 ## Reading data and handling missing values
#The Python script analyzes a dataset from the "Morningstar - European Mutual Funds.csv" file, focusing on handling missing values.
#Columns with missing values exceeding a 30% threshold are dropped, and numerical values undergo imputation using either mean or median. 
#The decision between mean and median is contingent upon the proximity of their values; if the difference is less than 0.5, mean imputation 
#is applied; otherwise, median imputation is utilized. Information detailing imputed numerical values is displayed. Categorical values are imputed with 
#the mode. The script concludes by presenting summaries of the remaining categorical and numerical columns, along with the total count of remaining missing values.  


m=pd.read_csv("data/Morningstar - European Mutual Funds.csv")




threshold = 0.3  

missing_percentages = m.isnull().mean()


columns_to_drop = missing_percentages[missing_percentages > threshold].index


df = m.drop(columns=columns_to_drop)


numerical_cols = df.select_dtypes(include=['number']).columns

for col in numerical_cols:
    if df[col].isnull().any():
        mean_val = df[col].mean()
        median_val = df[col].median()
        
        if abs(mean_val - median_val) < 0.5:  
            imputation_value = mean_val
            imputation_method = 'mean'
        else:
            imputation_value = median_val
            imputation_method = 'median'
        
        df[col].fillna(imputation_value, inplace=True)
        
        print(f"Imputed missing values in '{col}' with {imputation_method} ({imputation_value:.3f}).")

#print(df)


print(len(df.select_dtypes(include=['object']).columns))
print(len(df.select_dtypes(include=['float64', 'int64']).columns))


categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    mode_value = df[col].mode()[0]  
    df[col].fillna(mode_value, inplace=True)
    
print(df.isnull().sum().sum())





#%% [markdown]
## Feature selection using mutual information and pearson correlation for categorical and continuous columns respectively
#The Python script performs feature selection based on mutual information scores for categorical features and correlation
#  coefficients for numerical features concerning the target column 'fund_return_2019'. Categorical features with scores exceeding a 0.05 
# threshold are selected, and their names are printed. Additionally, numerical features with absolute correlation coefficients equal to or 
# greater than 0.1 are chosen and printed. The script ensures the target column's presence in the DataFrame to prevent errors, offering a 
# comprehensive approach to selecting relevant features for subsequent analysis. Adjustments to the threshold values can be made according 
# to specific analytical requirements.


from sklearn.feature_selection import mutual_info_regression
target_column = 'fund_return_2019'
if target_column in df.columns:
    
    numeric_columns = df.select_dtypes(include='number').columns
    categorical_columns = df.select_dtypes(include='object').columns

    
    all_columns = numeric_columns.union(categorical_columns)

    
    features = df[all_columns]

    
    cat_mi_scores = pd.Series(index=categorical_columns)

    for cat_col in categorical_columns:
        cat_mi_scores[cat_col] = mutual_info_regression(features[cat_col].astype('category').cat.codes.values.reshape(-1, 1), df[target_column])

    cat_mi_scores = cat_mi_scores.sort_values(ascending=False)

   
    features_to_select = 0.05
    selected_features = cat_mi_scores[cat_mi_scores >= features_to_select].index

    print("Selected Categorical Features:")
    print(selected_features)
else:
    print(f"Error: Target column '{target_column}' not found in DataFrame.")
    


if target_column in df.columns:
    numeric_columns = df.select_dtypes(include='number').columns

    features = df[numeric_columns]

    correlation_with_target = features.corrwith(df[target_column])

    selected_features = correlation_with_target[abs(correlation_with_target) >= 0.1]

    print("Selected Features with Correlation >= 0.1:")
    print(selected_features.index)
else:
    print(f"Error: Target column '{target_column}' not found in DataFrame.")
    
    
    
    

#%% [markdown]

## XGboost
#The provided Python script focuses on building and evaluating an XGBoost regression model for predicting 'fund_return_2019' 
# based on a selected set of features. The chosen features encompass various aspects such as asset allocation, involvement criteria, 
# financial metrics, and historical returns. Categorical columns are one-hot encoded, and the resulting features are combined with the 
# original numerical features. The dataset is split into training and testing sets, and an XGBoost regression model is trained on the 
# training set. The script evaluates the model on the test set, calculating the Mean Squared Error and R-squared as performance metrics. 
# This comprehensive approach aims to predict fund returns, leveraging a diverse set of features and advanced machine learning techniques. 
# Adjustments to hyperparameters and feature selection can be made for further refinement based on specific objectives.


columns_to_extract =['asset_stock', 'asset_bond', 'asset_cash', 'asset_other',
       'ongoing_cost', 'management_fees', 'involvement_abortive_contraceptive',
       'involvement_alcohol', 'involvement_animal_testing',
       'involvement_controversial_weapons', 'involvement_gambling',
       'involvement_gmo', 'involvement_military_contracting',
       'involvement_nuclear', 'involvement_palm_oil', 'involvement_pesticides',
       'involvement_small_arms', 'involvement_thermal_coal',
       'involvement_tobacco', 'shareclass_size', 'fund_size',
       'fund_trailing_return_ytd', 'fund_trailing_return_3years',
       'nav_per_share','fund_return_2018_q4', 'fund_return_2018_q3', 'fund_return_2018_q2',
       'fund_return_2018_q1', 'fund_return_2017_q4', 'fund_return_2017_q3',
       'fund_return_2017_q2', 'fund_return_2017_q1', 'quarters_up',
       'quarters_down','inception_date', 'category',
       'investment_strategy', 'investment_managers', 'fund_benchmark',
       'morningstar_benchmark', 'country_exposure', 'latest_nav_date',
       'nav_per_share_currency', 'shareclass_size_currency', 'fund_size_currency', 'top5_holdings', 'fund_return_2019']


df1 = df[columns_to_extract].copy()

#df1.head()


X = df1.drop('fund_return_2019', axis=1)
y = df1['fund_return_2019']


categorical_columns = X.select_dtypes(include=['object']).columns

encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_columns])

X_encoded = hstack([X.drop(columns=categorical_columns).values, X_encoded])

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',  
    'eval_metric': 'rmse',  
    'max_depth': 3,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

num_rounds = 100
model = xgb.train(params, dtrain, num_rounds)

y_pred = model.predict(dtest)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error on the test set: {mse:.2f}")
print(f"R-squared on the test set: {r2:.2%}")





#%% [markdown]

## Elastic net
#The provided Python script involves building and evaluating an Elastic Net regression model for predicting 'fund_return_2019' 
# based on a selected set of features. The chosen features encompass various aspects, including asset allocation, involvement criteria, 
# financial metrics, and historical returns. Categorical columns are one-hot encoded using the OneHotEncoder with sparse representation, 
# and the resulting features are combined with the original numerical features. The dataset is split into training and testing sets, and an 
# Elastic Net regression model is trained on the training set with specified hyperparameters (alpha=1.0 and l1_ratio=0.5). The script evaluates 
# the model on the test set, calculating the Mean Squared Error and R-squared as performance metrics. This approach utilizes regularization 
# techniques for enhanced predictive performance and allows for flexibility in controlling model complexity. Adjustments to hyperparameters 
# and feature selection can be made based on specific objectives.




X = df1.drop('fund_return_2019', axis=1)
y = df1['fund_return_2019']


categorical_columns = X.select_dtypes(include=['object']).columns

encoder = OneHotEncoder(sparse=True, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_columns])

X_encoded = hstack([X.drop(columns=categorical_columns).values, X_encoded])

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


alpha_value = 1.0  
l1_ratio_value = 0.5  
elastic_net_model = ElasticNet(alpha=alpha_value, l1_ratio=l1_ratio_value)
elastic_net_model.fit(X_train, y_train)


y_pred = elastic_net_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error on the test set: {mse:.2f}")
print(f"R-squared on the test set: {r2:.2%}")






#%% [markdown]

## XGBoost hyper parameter tuning
# The provided Python script involves tuning hyperparameters for an XGBoost regression model using Grid Search. The model aims to predict
#  'fund_return_2019' based on a selected set of features, covering aspects such as asset allocation, involvement criteria, financial metrics, 
# and historical returns. Categorical columns are one-hot encoded using the OneHotEncoder, and the resulting features are combined with 
# the original numerical features. The dataset is split into training and testing sets. Hyperparameter tuning is performed using a grid 
# of potential values for 'max_depth', 'learning_rate', and 'n_estimators'. The script employs 3-fold cross-validation to assess the 
# negative mean squared error as the scoring metric. The best hyperparameters are identified, and the model is trained with these optimal 
# settings. The script evaluates the tuned model on the test set, calculating the Mean Squared Error and R-squared as performance metrics. 
# This approach aims to enhance the model's predictive capabilities through systematic hyperparameter optimization.


X = df1.drop('fund_return_2019', axis=1)
y = df1['fund_return_2019']


categorical_columns = X.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_columns])


X_encoded = hstack([X.drop(columns=categorical_columns).values, X_encoded])


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


param_grid = {
    'max_depth': [3, 6],
    'learning_rate': [0.1, 0.2],
    'n_estimators': [50, 100]
}


xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', seed=42)


grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)


grid_search.fit(X_train, y_train)


best_params = grid_search.best_params_
print("Best Hyperparameters:")
print(best_params)

best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error on the test set: {mse:.2f}")
print(f"R-squared on the test set: {r2:.2%}")








models = [ 'XGB','XGB (After Tuning)', 'Elastic Net', 'RF',
          'RF(after tuning)']


mse_values = [14.26,6.90, 23.16, 6.78, 4.07]


r2_values = [82.25,91.42, 71.18, 91.55, 94.9]

results_df = pd.DataFrame({'Model': models, 'MSE': mse_values, 'R-squared': r2_values})


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

sns.barplot(x='Model', y='MSE', data=results_df, ax=axes[0])
axes[0].set_title('Mean Squared Error Comparison')


sns.barplot(x='Model', y='R-squared', data=results_df, ax=axes[1])
axes[1].set_title('R-squared Comparison')


plt.tight_layout()

plt.savefig('model_comparison.png')

plt.show()








# %%
