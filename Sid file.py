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



m=pd.read_csv("D:\\Morningstar - European Mutual Funds.csv")


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
