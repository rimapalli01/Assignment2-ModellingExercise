
"""

@author: rimapalli
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
features_df = pd.read_csv('data/features.csv')
stores_df = pd.read_csv('data/stores.csv')
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Print data
print(features_df.head())
print(stores_df.head())
print(train_df.head())
print(test_df.head())


# 1. Merge Data
train_merged = train_df.merge(stores_df, on='Store').merge(features_df, on=['Store', 'Date'], how='left')
test_merged = test_df.merge(stores_df, on='Store').merge(features_df, on=['Store', 'Date'], how='left')

# Confirming the columns in the merged DataFrame
print("Columns in train_merged:", train_merged.columns.tolist())
print("Columns in test_merged:", test_merged.columns.tolist())

# Check if 'IsHoliday_x' and 'IsHoliday_y' are identical
identical_holiday_columns = (train_merged['IsHoliday_x'] == train_merged['IsHoliday_y']).all()

print(f"IsHoliday_x and IsHoliday_y are identical: {identical_holiday_columns}")

# keep 'IsHoliday_x' and discard 'IsHoliday_y'.
train_merged = train_merged.drop(columns=['IsHoliday_y'])
train_merged = train_merged.rename(columns={'IsHoliday_x': 'IsHoliday'})

test_merged = test_merged.drop(columns=['IsHoliday_y'])
test_merged = test_merged.rename(columns={'IsHoliday_x': 'IsHoliday'})

# 2. Handle Missing Values
# Fill missing numerical values with the median for columns present in both datasets
common_columns = set(train_merged.columns) & set(test_merged.columns)
numerical_columns = train_merged.select_dtypes(include=['float64', 'int64']).columns

for column in numerical_columns:
    if column in common_columns:
        median_value = train_merged[column].median()
        train_merged[column].fillna(median_value, inplace=True)
        test_merged[column].fillna(median_value, inplace=True)

if 'Type' in common_columns:
    mode_value = train_merged['Type'].mode()[0]
    train_merged['Type'].fillna(mode_value, inplace=True)
    test_merged['Type'].fillna(mode_value, inplace=True)

# 3. Feature Engineering
# Convert 'Date' to datetime and create 'Year', 'Month', 'Week', 'IsHolidayFlag' as features for train_merged
train_merged['Date'] = pd.to_datetime(train_merged['Date'])
train_merged['Year'] = train_merged['Date'].dt.year
train_merged['Month'] = train_merged['Date'].dt.month
train_merged['Week'] = train_merged['Date'].dt.isocalendar().week
train_merged['IsHolidayFlag'] = train_merged['IsHoliday'].astype(int)  # Convert boolean to int

# Convert 'Date' to datetime and create 'Year', 'Month', 'Week', 'IsHolidayFlag' as features for test_merged
test_merged['Date'] = pd.to_datetime(test_merged['Date'])
test_merged['Year'] = test_merged['Date'].dt.year
test_merged['Month'] = test_merged['Date'].dt.month
test_merged['Week'] = test_merged['Date'].dt.isocalendar().week
test_merged['IsHolidayFlag'] = test_merged['IsHoliday'].astype(int)

# Display the shapes of the datasets
print("Train shape:", train_merged.shape)
print("Test shape:", test_merged.shape)

# Display basic statistics
print(train_merged.describe())

# Check for remaining missing values
print(train_merged.isnull().sum())

# 4. Encode Categorical Variables
# Encode 'Type' as it is a categorical feature
label_encoder = LabelEncoder()
train_merged['Type'] = label_encoder.fit_transform(train_merged['Type'])
test_merged['Type'] = label_encoder.transform(test_merged['Type'])

# Prepare features and target variables
X = train_merged[['Store', 'Dept', 'Type', 'Size', 'Week', 'IsHolidayFlag', 'Year', 'Month']]
y = train_merged['Weekly_Sales']

# 5. Split the Data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data preprocessing and feature engineering completed.")

#Model Evaluation
# Function to evaluate model performance
def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    
    mse = mean_squared_error(y_val, predictions)
    rmse = np.sqrt(mse)  # Calculate RMSE
    mae = mean_absolute_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)
    
    print(f"Model Performance: {model.__class__.__name__}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")  # Display RMSE
    print(f"Mean Absolute Error: {mae}")
    print(f"R^2 Score: {r2}")
    print("-" * 50)
    
    return mse, rmse, mae, r2

# Instantiate the models with tuned hyperparameters
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=4, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)

# Evaluate Random Forest model
rf_scores = evaluate_model(rf_model, X_train, y_train, X_val, y_val)

# Evaluate Gradient Boosting model
gb_scores = evaluate_model(gb_model, X_train, y_train, X_val, y_val)

#Plotting the evaluation
# Train the RandomForestRegressor model and make predictions
rf_model.fit(X_train, y_train)
predictions_rf = rf_model.predict(X_val)

# Actual vs. Predicted Plot for RandomForestRegressor
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_val, y=predictions_rf, alpha=0.6)
sns.lineplot(x=y_val, y=y_val, color='red')  # Line for perfect predictions
plt.title('Actual vs. Predicted Values - RandomForestRegressor')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

# Residuals Plot for RandomForestRegressor
residuals_rf = y_val - predictions_rf
plt.figure(figsize=(10, 6))
sns.histplot(residuals_rf, bins=30, kde=True)
plt.title('Residuals Distribution - RandomForestRegressor')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Feature Importance Plot for RandomForestRegressor
feature_importance_rf = rf_model.feature_importances_
sorted_idx_rf = np.argsort(feature_importance_rf)[::-1]
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance_rf[sorted_idx_rf], y=X_train.columns[sorted_idx_rf])
plt.title('Feature Importance - RandomForestRegressor')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.show()

# Train the GradientBoostingRegressor model and make predictions
gb_model.fit(X_train, y_train)
predictions_gb = gb_model.predict(X_val)

# Actual vs. Predicted Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_val, y=predictions_gb, alpha=0.6)
sns.lineplot(x=y_val, y=y_val, color='red')  # Line for perfect predictions
plt.title('Actual vs. Predicted Values - GradientBoostingRegressor')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

# Residuals Plot
residuals_gb = y_val - predictions_gb
plt.figure(figsize=(10, 6))
sns.histplot(residuals_gb, bins=30, kde=True)
plt.title('Residuals Distribution - GradientBoostingRegressor')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Feature Importance Plot for GradientBoostingRegressor
feature_importance_gb = gb_model.feature_importances_
sorted_idx_gb = np.argsort(feature_importance_gb)[::-1]
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance_gb[sorted_idx_gb], y=X_train.columns[sorted_idx_gb])
plt.title('Feature Importance - GradientBoostingRegressor')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.show()

#Comparison of two models
metrics = ['MSE', 'RMSE', 'MAE', 'R^2']

# Determining the better model for each metric (RF=1, GB=-1, Tie=0)
comparison_scores = [1 if rf < gb else -1 if gb < rf else 0 for rf, gb in zip(rf_scores[:-1], gb_scores[:-1])] + \
                    [1 if rf > gb else -1 if gb > rf else 0 for rf, gb in zip(rf_scores[-1:], gb_scores[-1:])]

# Counting wins for each model
rf_wins = comparison_scores.count(1)
gb_wins = comparison_scores.count(-1)

# Best model determination
best_model = "Random Forest" if rf_wins > gb_wins else "Gradient Boosting" if gb_wins > rf_wins else "Tie"

# Plotting
fig, ax = plt.subplots()
indices = np.arange(len(metrics))
width = 0.35

rf_bars = ax.bar(indices - width/2, rf_scores, width, label='Random Forest', color='skyblue')
gb_bars = ax.bar(indices + width/2, gb_scores, width, label='Gradient Boosting', color='orange')

# Labels and Title
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Model Performance Metrics Comparison\nOverall Best Model: ' + best_model)
ax.set_xticks(indices)
ax.set_xticklabels(metrics)
ax.legend(loc='upper left')

plt.show()

