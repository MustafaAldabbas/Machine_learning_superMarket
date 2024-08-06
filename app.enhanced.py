import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

# Load the dataset
file_path = '/Users/mustafaaldabbas/Documents/GitHub/Machine_learning_superstore/df_cleaned.csv'
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_predictions))

rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

gb_model = GradientBoostingRegressor(n_estimators=100)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))

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

# Streamlit app
st.set_page_config(page_title="Retail Sales Forecasting", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
pages = st.sidebar.radio("Go to", ["Introduction", "EDA", "Modeling", "Forecasting", "Conclusion"])

st.sidebar.markdown("### Authors")
st.sidebar.image('/Users/mustafaaldabbas/Documents/GitHub/Machine_learning_superstore/my pic/Mustafa HS2.jpg', width=200)
st.sidebar.markdown("Mustafa Adabbas")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/mustafa-aldabbas-85256b95/)")
st.sidebar.image('/Users/mustafaaldabbas/Documents/GitHub/Machine_learning_superstore/my pic/cv photo .jpg', width=200)
st.sidebar.markdown("Natalia Gravereuaux")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/nmikh/)")

# Download button function
def download_button(df, filename, label):
    csv = df.to_csv(index=False)
    st.download_button(label, csv, filename, "text/csv")

# Introduction page
if pages == "Introduction":
    st.title("Retail Sales Forecasting 📈")
    st.markdown("""
    ## Introduction
    Welcome to the Retail Sales Forecasting project! This project aims to forecast the sales for the next 7 days using historical sales data from a global superstore. We will explore various machine learning models and select the best-performing model based on evaluation metrics.
    
    ### Dataset
    The dataset contains retail sales data from a global superstore over four years. The data includes:
    - **Date:** The date of the sale.
    - **Total:** The total sales amount on that date.
    
    ### Objective
    The goal is to predict future sales based on historical data, helping the store to manage inventory and optimize sales strategies.
    """)
    st.write("Here are the first few rows of the dataset:")
    st.dataframe(df.head())
    download_button(df, "sales_data.csv", "Download Dataset")

# EDA page
elif pages == "EDA":
    st.title("Exploratory Data Analysis (EDA) 📊")
    st.markdown("## Data Visualization")
    st.write("Total Sales Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sales_by_date['date'], sales_by_date['total'], marker='o')
    ax.set_title('Total Sales Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Sales')
    ax.grid(True)
    st.pyplot(fig)
    download_button(sales_by_date, "sales_by_date.csv", "Download Aggregated Data")

# Modeling page
elif pages == "Modeling":
    st.title("Modeling 🧠")
    st.markdown("## Model Development and Evaluation")
    st.write("We developed and evaluated multiple machine learning models to forecast sales.")

    tab1, tab2, tab3, tab4 = st.tabs(["Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting Regressor", "K-Nearest Neighbors Regressor"])

    with tab1:
        st.subheader("Decision Tree Regressor")
        st.write("Root Mean Squared Error (RMSE):", dt_rmse)
        st.markdown("A Decision Tree Regressor splits the data into subsets based on feature values and makes decisions at each node. It tends to overfit the data, capturing noise and fluctuations.")

    with tab2:
        st.subheader("Random Forest Regressor")
        st.write("Root Mean Squared Error (RMSE):", rf_rmse)
        st.markdown("A Random Forest Regressor is an ensemble method that combines multiple decision trees to improve accuracy and control over-fitting. It provides better performance by averaging the results of multiple trees.")

    with tab3:
        st.subheader("Gradient Boosting Regressor")
        st.write("Root Mean Squared Error (RMSE):", gb_rmse)
        st.markdown("Gradient Boosting Regressor is an ensemble technique that builds models sequentially, each correcting the errors of its predecessor. It can sometimes overfit if not properly tuned.")

    with tab4:
        st.subheader("K-Nearest Neighbors Regressor")
        st.write("Root Mean Squared Error (RMSE):", knn_rmse)
        st.markdown("K-Nearest Neighbors (KNN) Regressor predicts the target value based on the average of the nearest neighbors' target values. It provided the best performance in our case.")
    download_button(df, "modeling_data.csv", "Download Modeling Data")

# Forecasting page
elif pages == "Forecasting":
    st.title("Forecasting 🔮")
    st.markdown("## Sales Forecast for the Next 7 Days")
    st.write("Using the K-Nearest Neighbors Regressor, we forecasted sales for the next 7 days.")
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
    download_button(forecast_features, "forecasted_sales.csv", "Download Forecast Data")

# Conclusion page
elif pages == "Conclusion":
    st.title("Conclusion 🏁")
    st.markdown("""
    ## Conclusion
    In this project, we explored various machine learning models to forecast retail sales for a global superstore. After evaluating multiple models, we found that the K-Nearest Neighbors Regressor provided the best performance with the lowest RMSE.
    
    ### Key Takeaways
    - **Data Preparation:** Aggregating sales data by date and extracting relevant features (year, month, day) is crucial for model performance.
    - **Model Selection:** Evaluating multiple models helps in selecting the best-performing one. In our case, the KNN Regressor outperformed others.
    - **Hyperparameter Tuning:** Fine-tuning model parameters can significantly improve performance.
    - **Forecasting:** Accurate sales forecasting can help retail stores in managing inventory and optimizing sales strategies.
    
    ### Future Work
    - **Model Improvement:** Further tuning and experimenting with additional models could improve performance.
    - **Additional Features:** Including more features such as promotions, holidays, and competitor prices could enhance the model.
    - **Longer Forecasting:** Extending the forecasting period beyond 7 days.

    Thank you for exploring this project with us! 🙌
    """)
    download_button(forecast_features, "forecasted_sales.csv", "Download Forecast Data")
