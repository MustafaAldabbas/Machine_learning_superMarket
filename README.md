
# Retail Sales Forecasting
## Author 
* Mustafa Aldabbas, connect with me on [LinkedIn](https://www.linkedin.com/in/mustafa-aldabbas-85256b95/), [on X](https://x.com/Mustafa_dabbas)
* Natalia Gravereaux

## Project Overview

This project aims to perform time series analysis and predict the sales for the next 7 days using retail data from a global superstore. We applied various machine learning models to achieve this objective, including K-Nearest Neighbors, Decision Tree, Random Forest, and Gradient Boosting.

## Data Selection and Preparation

The dataset used in this project includes retail sales data from a global superstore over four years. The data was cleaned and preprocessed to facilitate analysis and modeling.

## Feature Engineering and Selection

We extracted features such as year, month, and day from the date column to be used in the regression models.

## Model Building and Evaluation

We experimented with several models:
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **K-Nearest Neighbors Regressor**

### Model Performance

The performance of each model was evaluated using RMSE (Root Mean Squared Error):

- **Decision Tree Regressor RMSE:** 1843.44
- **Random Forest Regressor RMSE:** 1717.29
- **Gradient Boosting Regressor RMSE:** 1993.27
- **K-Nearest Neighbors Regressor RMSE:** 1676.72

### Hyperparameter Tuning

We performed hyperparameter tuning to optimize the models:

- **Random Forest Regressor:**
  - Best Parameters: `{'n_estimators': 200, 'max_depth': 20}`
  - Best RMSE: 1698.31

- **Gradient Boosting Regressor:**
  - Best Parameters: `{'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}`
  - Best RMSE: 1852.77

## Forecasting

Using the K-Nearest Neighbors Regressor, which showed the best performance, we forecasted sales for the next 7 days.

## Streamlit App

We developed a Streamlit app to present the project interactively. The app includes sections for data visualization, model training and evaluation, and forecasting.

## Key Findings and Insights

- **Best Model:** K-Nearest Neighbors Regressor
- **Key Features:** Year, month, and day extracted from the date column were effective features for predicting sales.

## Real-World Application and Impact

This model can help retail stores forecast future sales, enabling better inventory management and sales strategies.

## Challenges and Learnings

- **Data Cleaning:** Handling missing values and outliers was crucial.
- **Feature Engineering:** Extracting relevant features from the date column improved model performance.
- **Model Selection:** Evaluating multiple models helped in selecting the best-performing one.

## Future Work and Improvements

- **Model Improvement:** Further tuning and experimenting with additional models could improve performance.
- **Additional Features:** Including more features such as promotions, holidays, and competitor prices could enhance the model.
- **Longer Forecasting:** Extending the forecasting period beyond 7 days.

## How to Run the Streamlit App

1. Clone the repository.
2. Install the required libraries from `requirements.txt`.
3. Run the app using the command:
   ```bash
   streamlit run app.py
   ```

## Authors

- [Your Name]

## License

This project is licensed under the MIT License.
