import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot 
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"cement.csv")
df.info()
df.columns
df.drop(columns={'month', 'GDP_Construction_Rs_Crs', 'GDP_Real Estate_Rs_Crs',
       'Oveall_GDP_Growth%', 'water_source', 'limestone', 'Coal',
       'Home_Interest_Rate', 'Trasportation_Cost', 'Population', 'order',
       'unit_price', 'Total_Price'},inplace =True)

Train = df.head(85)
Test = df.tail(12)

df1 = pd.read_csv('test_arima.csv', index_col = 0)


tsa_plots.plot_acf(df.sale, lags = 12)
tsa_plots.plot_pacf(df.sale, lags = 12)

# ARIMA with AR = 12, MA = 6
model1 = ARIMA(Train.sale, order = (12, 1, 6))
res1 = model1.fit()
print(res1.summary())

# Forecast for next 12 months
start_index = len(Train)
start_index
end_index = start_index + 11
forecast_test = res1.predict(start = start_index, end = end_index)

print (forecast_test)

# Evaluate forecasts
rmse_test = sqrt(mean_squared_error(Test.sale, forecast_test))
print('Test RMSE: %.3f' % rmse_test)

# plot forecasts against actual outcomes
pyplot.plot(Test.sale)
pyplot.plot(forecast_test, color = 'red')
pyplot.show()

import pmdarima as pm

ar_model = pm.auto_arima(Train.sale, start_p = 0, start_q = 0,
                      max_p = 16, max_q = 16, # maximum p and q
                      m = 1,              # frequency of series
                      d = None,           # let model determine 'd'
                      seasonal = False,   # No Seasonality
                      start_P = 0, trace = True,
                      error_action = 'warn', stepwise = True)

model = ARIMA(Train.sale, order = (3,1,5))
res = model.fit()
print(res.summary())

# Forecast for next 12 months
start_index = len(Train)
end_index = start_index + 11
forecast_best = res.predict(start = start_index, end = end_index)

print(forecast_best)

rmse_best = sqrt(mean_squared_error(Test.sale, forecast_best))
print('Test RMSE: %.3f' % rmse_best)

pyplot.plot(Test.sale)
pyplot.plot(forecast_best, color = 'red')
pyplot.show()

print('Test RMSE with Auto-ARIMA: %.3f' % rmse_best)
print('Test RMSE with out Auto-ARIMA: %.3f' % rmse_test)

res1.save("model.pickle")
from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model.pickle")

start_index = len(df)
end_index = start_index + 11
forecast = model.predict(start = start_index, end = end_index)

print(forecast)