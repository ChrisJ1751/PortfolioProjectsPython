import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import datetime

# Load data
df = pd.read_csv(r"/Users/christopherjones/Documents/sales_data.csv")

# Group data by state and person
grouped_data = df.groupby(['State', 'Account Manager'])

# Define features and target variable
X = df['Date of Sale'].to_numpy().reshape(-1, 1)
y = df['Revenue'].values.reshape(-1, 1)

# Train model for each state and person
models = {}
for (state, account_manager), data in grouped_data:
    # Split data into training and testing sets
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Define features and target variable for this state and person
    X_train = (pd.to_datetime(train_data['Date of Sale']) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    y_train = train_data['Revenue'].values.reshape(-1, 1)
    X_test = pd.to_datetime(test_data['Date of Sale'])
    X_test = (X_test - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    X_test = X_test.values.reshape(-1,1)
    y_test = test_data['Revenue'].values.reshape(-1, 1)
    X_train = X_train.values.reshape(-1,1)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('State:', state)
    print('Account Manager:', account_manager)
    print('Mean squared error:', mse)
    print('R2 score:', r2)

    # Test Forest Against Itself

    linear_test_preds = model.predict(X_test)
    MAE_1 = mean_absolute_error(linear_test_preds, y_test)
    print('MAE_1:', MAE_1)

    # Validation RandomForestRegrssor

    rf = RandomForestRegressor(n_estimators=21, max_depth=9).fit(X_train, y_train)
    rf_train_preds = rf.predict(X_train)
    rf_test_preds = rf.predict(X_test)
    mae = mean_absolute_error(rf_train_preds, y_train), mean_absolute_error(rf_test_preds, y_test)
    print('Mean Abosulte Error:', mae)

    #Select Model

    X_val, X_hold, y_val, y_hold = train_test_split(X_test, y_test, test_size=0.5)
    X_val.shape, X_hold.shape, y_val.shape, y_hold.shape
    linear_val_preds = model.predict(X_val)
    MAE_2 = mean_absolute_error(y_train, X_train), mean_absolute_error(y_val, linear_val_preds)
    print('MAE_2', MAE_2)
    rf_val_preds = rf.predict(X_val)
    MAE_3 = mean_absolute_error(y_train, rf_train_preds), mean_absolute_error(y_val, rf_val_preds)
    print('MAE_3', MAE_3)
 
    # Add model to dictionary of models
    key = f"{state}_{account_manager}"
    models[key] = model

# Make sales forecasts for each state and person
future_dates = pd.date_range(start='2023-01-01', end='2023-12-31').ravel()
future_dates_numeric = (pd.to_datetime(future_dates) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
future_dates_numeric = future_dates_numeric.to_numpy().reshape(-1, 1)
forecast_data = {'State': [], 'Account Manager': [], 'Date of Sale': [], 'Revenue': []}
for key, value in models.items():
    # Generate forecasts for future dates
    future_revenue = value.predict(future_dates_numeric)
    
    # Add forecast data to dictionary
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
