import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import schedule
import time
import datetime
import random
import logging


# Set up logging
logging.basicConfig(filename='sales.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')
def run_AI_program():
    # Load data, Check to make sure the data loaded in correctly
    try:
        df = pd.read_csv(r"/Users/christopherjones/Documents/sales_data.csv")
    except FileNotFoundError:
        logging.error("Error: coild not find sales_data.csv file")
        return 
    logging.info("Sales data loaded succesfully")
    logging.info(f"Number of rows: {len(df)}")

    # Group data by state and account manager
    grouped_data = df.groupby(['State', 'Account Manager'])

    # Initialize dictionary to store models and performance metrics
    models = {}

    # Loop over each unique combination of state and account manager
    for (state, account_manager), data in grouped_data:

        # Split data into training, validation, and test sets
        X_train_val = (pd.to_datetime(data['Date of Sale']) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        X_train_val = X_train_val.values.reshape(-1, 1)
        y_train_val = data['Revenue'].values.reshape(-1, 1)
        X_train_val, X_test, y_train_val, y_test = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
        X_train = X_train.reshape(-1, 1)
        X_val = X_val.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)

        # Train linear regression model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        # Evaluate linear regression model performance
        lr_train_preds = lr_model.predict(X_train)
        lr_val_preds = lr_model.predict(X_val)
        lr_test_preds = lr_model.predict(X_test)
        lr_metrics = {'Train MAE': mean_absolute_error(y_train, lr_train_preds),
                    'Val MAE': mean_absolute_error(y_val, lr_val_preds),
                    'Test MAE': mean_absolute_error(y_test, lr_test_preds),
                    'Train R^2': lr_model.score(X_train, y_train),
                    'Val R^2': lr_model.score(X_val, y_val),
                    'Test R^2': lr_model.score(X_test, y_test)}

        # Train random forest regression model
        rf_model = RandomForestRegressor(n_estimators=21, max_depth=9)
        rf_model.fit(X_train, y_train.ravel())

        # Evaluate random forest regression model performance
        rf_train_preds = rf_model.predict(X_train)
        rf_val_preds = rf_model.predict(X_val)
        rf_test_preds = rf_model.predict(X_test)
        rf_metrics = {'Train MAE': mean_absolute_error(y_train, rf_train_preds),
                    'Val MAE': mean_absolute_error(y_val, rf_val_preds),
                    'Test MAE': mean_absolute_error(y_test, rf_test_preds),
                    'Train R^2': rf_model.score(X_train, y_train),
                    'Val R^2': rf_model.score(X_val, y_val),
                    'Test R^2': rf_model.score(X_test, y_test)}

    
        # Select the better performing model
        if lr_metrics['Val R^2'] > rf_metrics['Val R^2']:
            best_model_type = 'Linear Regression'
            best_model = lr_model
            best_metrics = lr_metrics
        else:
            best_model_type = 'Random Forest Regression'
            best_model = rf_model
            best_metrics = rf_metrics
        
        # Store best model and performance metrics in dictionary
        models[(state, account_manager)] = {'Best Model': best_model, 'Best Model Type': best_model_type, 'Best Metrics': best_metrics}

                    # Log results for this state and account manager
        logging.info(f'State: {state}, Account Manager: {account_manager}')
        for model_type in ['Linear Regression', 'Random Forest Regression']:
            logging.info(f'{model_type} Metrics:')
            try:
                if model_type in models[(state, account_manager)]:
                    for metric_name, metric_value in models[(state, account_manager)][model_type]['Metrics'].items():
                        logging.info(f'{metric_name}: {metric_value:.4f}')
                else:
                    logging.info(f'No metrics found for {model_type} model')
            except KeyError:
                logging.info(f'No metrics found for {model_type} model')
        
        # Define forecast data
        forecast_dates = pd.date_range(start='2023-03-21', end='2023-04-20')
        forecast_state = ['California', 'New York', 'Texas', 'Florida']
        forecast_account_manager = ['John Smith', 'Bill Johnson', 'Jessica Pearson', 'Mike Anderson']
        forecast_data = pd.DataFrame({'Date of Sale': forecast_dates.repeat(len(forecast_state) * len(forecast_account_manager)),
                              'State': np.tile(np.repeat(forecast_state, len(forecast_dates)), len(forecast_account_manager)),
                              'Account Manager': np.tile(np.repeat(forecast_account_manager, len(forecast_dates)), len(forecast_state))})

        forecasts = generate_sales_forecasts(models, forecast_data)

# Make sales forecasts for each state and person
def generate_sales_forecasts(models, forecast_data):
    forecasts = {}

    # Loop over each unique combination of state and account manager
    for (state, account_manager), model_data in models.items():

        # Make predictions with the best model for this state and account manager
        best_model = model_data['Best Model']
        forecast_preds = best_model.predict(forecast_data)

        # Store forecast results for this state and account manager
        forecasts[(state, account_manager)] = {'Forecast Dates': forecast_data.index.tolist(),
                                            'Forecast Revenue': forecast_preds.tolist()}

    return forecasts

    #Add forecast data to dictionary
    for i in range(len(future_dates)):
        forecast_data['State'].append(key.split('_')[0])
        forecast_data['Account Manager'].append(key.split('_')[1])
        forecast_data['Date of Sale'].append(future_dates[i])
        forecast_data['Revenue'].append(future_revenue[i][0])


    # Convert forecast data to DataFrame
    forecast_df = pd.DataFrame(forecast_data)


    # Save Forecast data to a CSV file
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'sales_forecasts_{timestamp}.csv'
    forecast_df.to_csv(filename, index=False)


# Scheduling the program to run every day
schedule.every(1).minute.do(run_AI_program)


while True:
   schedule.run_pending()
   time.sleep(1)