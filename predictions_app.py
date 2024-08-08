import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the datasets
file_path_main = '/Users/mustafaaldabbas/Documents/GitHub/Machine_learning_superstore/Dataset/Mustafa Datasets /df_cleaned.csv'
file_path_combined = '/Users/mustafaaldabbas/Documents/GitHub/Machine_learning_superstore/Dataset/Mustafa Datasets /combined_df.csv'
file_path_encoded = '/Users/mustafaaldabbas/Documents/GitHub/Machine_learning_superstore/Dataset/Mustafa Datasets /combined_df_encoded.csv'
file_path_models = '/Users/mustafaaldabbas/Documents/GitHub/Machine_learning_superstore/Dataset/Mustafa Datasets /results_of_models.csv'
file_path_predictions = '/Users/mustafaaldabbas/Documents/GitHub/Machine_learning_superstore/Dataset/Mustafa Datasets /future_predictions.csv'

df = pd.read_csv(file_path_main)
df_combined = pd.read_csv(file_path_combined)
df_encoded = pd.read_csv(file_path_encoded)
df_models = pd.read_csv(file_path_models)
df_predictions = pd.read_csv(file_path_predictions)

# Convert the date column to datetime format
df['date'] = pd.to_datetime(df['date'])
df_predictions['date'] = pd.to_datetime(df_predictions['date'])

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

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))

ada_model = AdaBoostRegressor()
ada_model.fit(X_train, y_train)
ada_predictions = ada_model.predict(X_test)
ada_rmse = np.sqrt(mean_squared_error(y_test, ada_predictions))

# Streamlit app
st.set_page_config(page_title="Retail Sales Forecasting", layout="wide")

# Sidebar navigation
st.sidebar.title("Sales Predictions Project")
pages = st.sidebar.radio("Go to", ["Introduction", "Workflow & Objectives", "EDA", "Modeling", "Forecasting", "Achievement and conclusion",])

# Authors section in sidebar
st.sidebar.markdown("### Authors")
st.sidebar.image('/Users/mustafaaldabbas/Documents/GitHub/Machine_learning_superstore/my pic/Mustafa HS2.jpg', width=100)
st.sidebar.markdown("Mustafa Aldabbas")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/your-linkedin-id/)")
st.sidebar.image('/Users/mustafaaldabbas/Documents/GitHub/Machine_learning_superstore/my pic/cv photo .jpg', width=100)
st.sidebar.markdown("Natalia Gravereaux")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/another-linkedin-id/)")

# Introduction page
if pages == "Introduction":
    st.markdown('<h1 style="color: LightBlue;">Retail Sales Forecasting 📈 </h1>', unsafe_allow_html=True)
    st.image ('/Users/mustafaaldabbas/Documents/GitHub/Machine_learning_superstore/my pic/retail forcasting .png', width=1000)
    st.markdown("""
    ## Introduction
    ##### **Welcome to the Retail Sales Forecasting project!**
    This project aims to forecast the sales for the next 7 days using historical sales data from a global superstore.<br>
    We will explore various machine learning models and select the best-performing model based on evaluation metrics.<br>
    
    ### **Dataset**
    The dataset contains retail sales data from a global superstore over 3 months from 01.01.2019 to 30.03.2019.<br> The data includes:<br>
    - **Date:** The date of the sale.<br>
    - **Total:** The total sales amount on that date.<br>
    - **Branch:** 3 different branches of the superstore.<br>
    - **Customer type:** are they normal or member customer.<br>
    - **Productline:** Like health, elcetronics, home and lifestyle.<br>
    - **Unit price:** the price of each unit purchsed.<br>
    - **quantity:** The numer of purchesed articles<br>
    - **Gross:** Total gross<br>
   
    """, unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Original Data", "Combined Data", "Encoded Data"])
    with tab1:
        st.dataframe(df.head())
    with tab2:
        st.dataframe(df_combined.head())
    with tab3:
        st.dataframe(df_encoded.head())
 
# Project Goals page
elif pages == "Workflow & Objectives":
    st.title("Workflow and Objectives 🎯")
    st.image ('/Users/mustafaaldabbas/Documents/GitHub/Machine_learning_superstore/my pic/workflow .jpg', width=1000)
    st.markdown("""
    ## Workflow 
    In this project, we will cover several key steps in preparing data for machine learning and employing various<br> predictive models to forecast future supermarket sales. Our main objectives include:<br>
    - **Data Preparation:** Preparing the data for machine learning.<br>
    - **Exploratory Data Analysis (EDA):** Conducting EDA to understand the data.<br>
    - **Feature Engineering:** Applying techniques like lag, moving average, and one-hot encoding.<br>
    - **Data Transformation and Normalization:** Transforming and normalizing the data.<br>
    - **Model Implementation:** Using different models to predict future sales of the supermarket, including:<br>
      - Linear Regression<br>
      - Decision Tree Regressor<br>
      - Random Forest Regressor<br>
      - Ensemble Methods: Gradient Boosting Regressor<br>
      - CatBoost<br>
      - K-Nearest Neighbors Regressor<br>
      - Ensemble Methods: LightGBM<br>
      - Ensemble Methods: XGBoost<br>
      - ARIMA (AutoRegressive Integrated Moving Average)<br>
    - **Hyperparameter Tuning:** Tuning hyperparameters for optimal model performance.<br>
    - **Model Training and Testing:** Training the models and testing their performance.<br>
    - **Model Selection and Prediction:** Choosing the best model and predicting future sales.
    ## **Objective**
    - predict one week of future sales based on the historial data
    - Predict one week oof future gross income based on the historical data
    - helping the store to manage inventory and optimize sales strategies.
    """, unsafe_allow_html=True)

# EDA page
elif pages == "EDA":
    st.title("Exploratory Data Analysis (EDA) 📊")
    st.markdown("## Data Visualization")
    ## Exploratory Data Analysis EDA
    st.markdown("""
    - #### **Plot the total sales over time to identify any trends or patterns.**
    - #### **Perform a seasonal decomposition to understand the trend, seasonality, and residual components.**
   
   
    """, unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sales_by_date['date'], sales_by_date['total'], marker='o')
    ax.set_title('Total Sales Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Sales')
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("#### Seasonal Decomposition")
    decomposition = seasonal_decompose(sales_by_date['total'], period=30)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
    decomposition.observed.plot(ax=ax1, title='Observed')
    decomposition.trend.plot(ax=ax2, title='Trend')
    decomposition.seasonal.plot(ax=ax3, title='Seasonal')
    decomposition.resid.plot(ax=ax4, title='Residual')
    plt.tight_layout()
    st.pyplot(fig)

# Modeling page
elif pages == "Modeling":
    st.title("Modeling 🧠")
    st.markdown("## Model Development and Evaluation")
    st.image ('/Users/mustafaaldabbas/Documents/GitHub/Machine_learning_superstore/my pic/modeling .jpeg', width=1000)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Decision Tree", "Random Forest", "Gradient Boosting", "K-Nearest Neighbors", "Linear Regression", "AdaBoost"])

    with tab1:
        st.write("(RMSE) for Decision Tree Regressor:", dt_rmse)
    with tab2:
        st.write("(RMSE) for Random Forest Regressor:", rf_rmse)
    with tab3:
        st.write("(RMSE) for Gradient Boosting Regressor:", gb_rmse)
    with tab4:
        st.write("(RMSE) for K-Nearest Neighbors Regressor:", knn_rmse)
    with tab5:
        st.write("(RMSE) for Linear Regression:", lr_rmse)
    with tab6:
        st.write("(RMSE) for AdaBoost Regressor:", ada_rmse)

    st.markdown("## Model Results")
    st.dataframe(df_models)

# Forecasting page
elif pages == "Forecasting":
    st.title("Forecasting 🔮")
    st.markdown("## Sales Forecast for the Next 7 Days")
    st.dataframe(df_predictions)

    st.markdown("### Future predictions approach 1")
    st.image ('/Users/mustafaaldabbas/Documents/GitHub/Machine_learning_superstore/my pic/future prediction approach 2.png', width=1000)
    st.markdown("### Future predictions approach 2")
    st.image ('/Users/mustafaaldabbas/Documents/GitHub/Machine_learning_superstore/my pic/predictions_future.png', width=1000)
    st.markdown("### Gross income predictions ")
    st.image ('/Users/mustafaaldabbas/Documents/GitHub/Machine_learning_superstore/my pic/viz_gross_income_per_day.png', width=1000)
    
    

   

# Conclusion page
elif pages == "Achievement and conclusion":
    st.title("Achievements 🏆")
    st.image ('/Users/mustafaaldabbas/Documents/GitHub/Machine_learning_superstore/my pic/we didt it finally.png', width=1000)
    st.markdown("""
    #### In this project, we have achieved the following:
    - ##### Sales Prediction: Predicted the total sales of the supermarket for 7 days using two different approaches with different feature sets, applied across all models.
    - ##### Gross Income Prediction: Predicted the gross income for the supermarket for 7 days.
    """, unsafe_allow_html=True)
    st.markdown('<h1 style="color: Red;">Conclusion 🏁</h1>', unsafe_allow_html=True)
    st.markdown(""" 
    - In this project, we explored various machine learning models to forecast retail sales for a global superstore.<br>
    - The K-Nearest , Random FOrest regressor and Neighbors Regressor provided the best performance with the lowest RMSE.
                ,
      """, unsafe_allow_html=True)
    
