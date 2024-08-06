import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Load the dataset
file_path = '/Users/mustafaaldabbas/Documents/GitHub/Machine_learning_superMarket/df_cleaned.csv'
df = pd.read_csv(file_path)

# Convert the date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Aggregate sales data by date
sales_by_date = df.groupby('date')['total'].sum().reset_index()

# Extract year, month, and day from the date column
sales_by_date['year'] = sales_by_date['date'].dt.year
sales_by_date['month'] = sales_by_date['date'].dt.month
sales_by_date['day'] = sales_by_date['date'].dt.day

# Prepare the data
X = sales_by_date[['year', 'month', 'day']]
y = sales_by_date['total']

# Split the data into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train and evaluate models

# Decision Tree Regressor
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_predictions))

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

# Grid Search for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}
grid_search_rf = GridSearchCV(RandomForestRegressor(), param_grid_rf, cv=5, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)
best_params_rf = grid_search_rf.best_params_
best_model_rf = grid_search_rf.best_estimator_
best_rmse_rf = np.sqrt(-grid_search_rf.best_score_)

# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))

# Random Search for Gradient Boosting
param_dist_gb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.2, 0.3]
}
random_search_gb = RandomizedSearchCV(GradientBoostingRegressor(), param_dist_gb, n_iter=10, cv=5, scoring='neg_mean_squared_error')
random_search_gb.fit(X_train, y_train)
best_params_gb = random_search_gb.best_params_
best_model_gb = random_search_gb.best_estimator_
best_rmse_gb = np.sqrt(-random_search_gb.best_score_)

# K-Nearest Neighbors Regressor
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_rmse = np.sqrt(mean_squared_error(y_test, knn_predictions))

# Forecast the next 7 days using the K-Nearest Neighbors Regressor
last_date = sales_by_date['date'].max()
forecast_dates = pd.date_range(start=last_date, periods=8, inclusive='right')
forecast_features = pd.DataFrame({
    'date': forecast_dates,
    'year': forecast_dates.year,
    'month': forecast_dates.month,
    'day': forecast_dates.day
})
forecast_features['total'] = knn_model.predict(forecast_features[['year', 'month', 'day']])

# Set up the Streamlit app
st.title("Retail Sales Forecasting")

# Introduction
st.header("Introduction")
st.write("""
This app provides a time series analysis and forecasting for retail sales data. The goal is to predict the sales for the next 7 days using various machine learning models.
""")

# Data Overview
st.header("Data Overview")
st.write("Here are the first few rows of the dataset:")
st.dataframe(df.head())

# Data Visualization
st.header("Data Visualization")
st.write("Total Sales Over Time")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(sales_by_date['date'], sales_by_date['total'], marker='o')
ax.set_title('Total Sales Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Total Sales')
ax.grid(True)
st.pyplot(fig)

# Model Training and Evaluation
st.header("Model Training and Evaluation")

# Display RMSE for each model
st.write("Decision Tree Regressor RMSE:", dt_rmse)
st.write("Random Forest Regressor RMSE:", rf_rmse)
st.write("Random Forest Regressor Best RMSE (Grid Search):", best_rmse_rf)
st.write("Gradient Boosting Regressor RMSE:", gb_rmse)
st.write("Gradient Boosting Regressor Best RMSE (Random Search):", best_rmse_gb)
st.write("K-Nearest Neighbors Regressor RMSE:", knn_rmse)

# Forecasting
st.header("Forecasting")
st.write("Forecasted sales for the next 7 days:")

st.dataframe(forecast_features[['date', 'total']])

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(sales_by_date['date'], sales_by_date['total'], label='Observed')
ax.plot(forecast_features['date'], forecast_features['total'], marker='o', color='red', label='Forecast')
ax.set_title('Sales Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Total Sales')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Summary and Conclusion
st.header("Summary and Conclusion")
st.write("""
This project demonstrates the use of machine learning models for time series analysis and forecasting of retail sales. Among the models evaluated, the K-Nearest Neighbors Regressor showed the best performance with an RMSE of 1676.72. The forecasted sales for the next 7 days are provided above.
""")
