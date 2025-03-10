import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
from pandas.tseries.offsets import DateOffset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()
scaler_2 = MinMaxScaler()

def calculate_metrics(real_values, predicted_values):
    """
    Function to calculate MAPE, RMSE, and MAE.
    """
    real_values = real_values.squeeze()
    predicted_values = predicted_values.squeeze()
    mape = (np.abs((real_values - predicted_values) / real_values) * 100).mean()
    rmse = np.sqrt(mean_squared_error(real_values, predicted_values))
    mae = mean_absolute_error(real_values, predicted_values)
    
    return mape, rmse, mae

# Reading the dataset
General_Model = pd.read_csv(
    r'C:\Users\diego\OneDrive\Área de Trabalho\Pós graduação\TCC\Projeto\Dados\Test_Data.csv',
    encoding="ISO-8859-1",
    sep=';',
    parse_dates=['Data'],   # "Data" = date column in the file
    index_col='Data',
    decimal=","
)

# Renaming columns from Portuguese to English
General_Model.rename(
    columns={
        'Dummy_Amstel': 'Dummy_Amstel',
        'Dummy_Bavaria': 'Dummy_Bavaria',
        'Dummy_Eisenbahn': 'Dummy_Eisenbahn',
        'Dummy_Estrella_Galicia': 'Dummy_Estrella_Galicia',
        'Dummy_Heineken': 'Dummy_Heineken',
        'Dummy_Kaiser': 'Dummy_Kaiser',
        'Dummy_Sol': 'Dummy_Sol',
        'Dummy_Tiger': 'Dummy_Tiger',
        'Desocupação': 'Unemployment',
        'IPCA': 'IPCA',
        'PMC H S A&B': 'Retail_HSAB',
        'PMC TOTAL': 'Retail_TOTAL',
        'Auxílio': 'Aid',
        'Carnaval': 'Carnival',
        'Temperatura': 'Temperature',
        'Volume': 'Volume'  # Already English, but listed for completeness
    },
    inplace=True
)

# Sorting by index (date)
General_Model = General_Model.sort_index()

# Scaling data (fixed the DataFrame reference)
General_Model = pd.DataFrame(
    scaler_2.fit_transform(General_Model),
    columns=General_Model.columns,
    index=General_Model.index
)

# Splitting the data into training and test sets
train = General_Model[General_Model.index <= '2022-12-31']
test = General_Model[(General_Model.index >= '2023-01-01') & (General_Model.index <= '2023-04-30')]

# Converting the indices to monthly Period type
General_Model.index = General_Model.index.to_period('M')
train.index = train.index.to_period('M')
test.index = test.index.to_period('M')

# Defining exogenous variables
exog_vars = train[
    [
        'Dummy_Amstel', 'Dummy_Bavaria', 'Dummy_Eisenbahn', 'Dummy_Estrella_Galicia',
        'Dummy_Heineken', 'Dummy_Kaiser', 'Dummy_Sol', 'Dummy_Tiger', 'Unemployment',
        'IPCA', 'Retail_HSAB', 'Retail_TOTAL', 'Aid', 'Carnival', 'Temperature'
    ]
].values

# Building the SARIMAX model
model = SARIMAX(
    np.log1p(train['Volume']),  # Applying log to the training data
    exog=train[
        [
            'Dummy_Amstel', 'Dummy_Bavaria', 'Dummy_Eisenbahn', 'Dummy_Estrella_Galicia',
            'Dummy_Heineken', 'Dummy_Kaiser', 'Dummy_Sol', 'Dummy_Tiger', 'Unemployment',
            'IPCA', 'Retail_HSAB', 'Retail_TOTAL', 'Aid', 'Carnival', 'Temperature'
        ]
    ],
    order=(0, 0, 0),           # Chosen by auto_arima
    seasonal_order=(0, 1, 0, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

model_fit = model.fit(disp=False)
model_fit.summary()

# Predicting values of Unemployment, Retail_TOTAL, and Retail_HSAB
vars_to_predict = ['Unemployment', 'Retail_HSAB', 'Retail_TOTAL']

# Removing Temperature from train and test
del train['Temperature']
del test['Temperature']

# For each variable to be predicted, we fit a linear regression
for var in vars_to_predict:
    X_train = train.drop(vars_to_predict, axis=1)
    y_train = train[var]
    X_test = test.drop(vars_to_predict, axis=1)

    # Fitting the linear regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Predicting values for the first four months of 2023
    test[var] = lr_model.predict(X_test)

# Dropping Temperature from test (fixed the drop call)
test = test.drop('Temperature', axis=1)

# Setting up comparison values
real_values = test['Volume']
historic_values = train['Volume']
historic_values.index = train['Volume'].index.to_timestamp()

# Scaling the 'Carnival' variable
train_scaled = scaler.fit_transform(train[['Carnival']])
test_scaled = scaler.transform(test[['Carnival']])

# Using auto_arima to find best model parameters
General_Model_Auto = pm.auto_arima(
    np.log1p(train['Volume']),
    train[['Carnival']],
    start_p=1, start_q=1,
    test='adf',
    max_p=5, max_q=5,
    m=12,
    start_P=0,
    seasonal=True,
    d=None,
    D=1,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)
General_Model_Auto.summary()

# Training the SARIMAX model with exogenous 'Carnival'
train.index = train.index.to_period('M')
model = SARIMAX(
    np.log1p(train['Volume']),
    train[['Carnival']],  # Applying log to the training data
    order=(2, 2, 0),      # Parameters chosen by auto_arima
    seasonal_order=(0, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

model_fit = model.fit(disp=False)

# Making the prediction (Jan-Apr 2023) using exogenous variables
test.index = test.index.to_timestamp()
fitted = model_fit.get_prediction(
    start=pd.to_datetime('2023-01-01'),
    end=pd.to_datetime('2023-04-30'),
    dynamic=False,
    full_results=True,
    exog=test_scaled
)

predicted_values = np.expm1(fitted.predicted_mean)
real_values.index = predicted_values.index

# Collecting metrics
mape, rmse, mae = calculate_metrics(real_values, predicted_values)
df_metrics = pd.DataFrame(columns=['MAPE', 'RMSE', 'MAE'])  # Ensure this DataFrame exists
df_metrics.loc['SARIMAX_CARNIVAL_Model'] = [mape, rmse, mae]

# Comparing against "Goals" model (Metas)
Goals = pd.read_csv(
    r'C:\Users\diego\OneDrive\Área de Trabalho\Pós graduação\TCC\Projeto\Dados\Target_demand.csv',
    encoding="ISO-8859-1",
    sep=';',
    parse_dates=['Data'],
    index_col='Data',
    decimal=","
)

# In Portuguese code, "Metas" stands for targets/goals
predicted_values_goals = Goals
predicted_values_goals.index = predicted_values_goals.index.to_timestamp()
predicted_values_goals.index = real_values.index

mape, rmse, mae = calculate_metrics(real_values, predicted_values)
df_metrics.loc['Goals_Model'] = [mape, rmse, mae]

# Comparing both with SARIMA (no exogenous variables)
sarima_model = pm.auto_arima(
    np.log1p(train['Volume']),
    start_p=1, start_q=1,
    test='adf',
    max_p=5, max_q=5,
    m=12,
    start_P=0,
    seasonal=True,
    d=None,
    D=1,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)
sarima_model.summary()

fitted_sarima = sarima_model.predict(
    start=pd.to_datetime('2023-01-01'),
    end=pd.to_datetime('2023-04-30')
)
fitted_sarima = fitted_sarima.drop(fitted_sarima.index[4:])  # Same slicing as original
predicted_values_SARIMA = np.expm1(fitted_sarima)
predicted_values_SARIMA.index = real_values.index

mape, rmse, mae = calculate_metrics(real_values, predicted_values)
df_metrics.loc['SARIMA_Model'] = [mape, rmse, mae]

# Plotting results
predicted_values.index = predicted_values.index.to_timestamp()
predicted_values_SARIMA.index = predicted_values_SARIMA.index.to_timestamp()
real_values.index = real_values.index.to_timestamp()

plt.figure(figsize=(10, 5))
plt.plot(real_values, label='Actual Volume')
plt.plot(predicted_values, color='red', label='SARIMAX')
plt.plot(predicted_values_goals, color='blue', label='Goals')
plt.plot(predicted_values_SARIMA, color='green', label='SARIMA')
plt.title('Actual Volume vs Forecasted Volume')
plt.xlabel('Time')
plt.ylabel('Volume')
plt.legend()
plt.show()

# Checking covariance
df = train[['Retail_TOTAL', 'Retail_HSAB']]
correlation_matrix = df.corr()
print(correlation_matrix)

# Adding an intercept column for VIF calculations
df['Intercept'] = 1

# Calculating the VIF for each variable
vif = pd.DataFrame()
vif["Variable"] = df.columns
vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

# Printing the results
print(vif)
