# import libraries
import pandas as pd
from datetime import datetime
from matplotlib import pyplot
from pandas import DataFrame
# use a ARIMA machine learning model 
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm

# function that called a parser that convert the type of date to datetime in python 
def parser(x):
	return datetime.strptime(x, '%m/%d/%Y')
# read the data file  csv file 

df = pd.read_csv('data_prices.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
df.index = df.index.to_period('M')
# show the Date and the Price
print(df)

# show first 5 rows in dataset
print(df.head())

#check if there is a null  value in database
print(df.isnull().sum())

# describe the datafram , show the maen value and std .. etc.
print(df.describe())

# plot the data and show there is a There is an ascending relationship between history and prices, 
# and there is a rise in prices from the beginning of 2020 until 2022
ax = df.plot(figsize=(12,6))
ax.set(title='Price by year 2017 to 2022', ylabel='Price')


# fit an ARIMA model and plot residual errors

"""
data segmentation Non-seasonal ARIMA models are generally denoted ARIMA(p,d,q)\
 where parameters p, d, and q are non-negative integers,
  p is the order (number of time lags) of the autoregressive model,
  d is the degree of differencing (the number of times the data have had past values subtracted), and 
  q is the order of the moving-average ..

"""
# use a ARIMA  model with  parameters p, d, and q 

model = sm.tsa.arima.ARIMA(df, order=(5,1,0))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())

# line plot of residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
# density plot of residuals
residuals.plot(kind='kde')
pyplot.show()
# summary stats of residuals  (error diffenrt between the real and predicted)
print(residuals.describe())


# evaluate an ARIMA model using a walk-forward validation


# split into train and test sets
X = df.values
print(X)

# size = int(len(X) * 0.66)
size = int(len(X) * 0.80)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list() # list 

# X real value Price
print(len(X))
print("size of total dataset that used : ",size)
print("size of train data : " , len(train))
print("size of test data : ",len(test))


# walk-forward validation
for t in range(len(test)):
	model = sm.tsa.arima.ARIMA(history, order=(5,1,0))
	model_fit = model.fit()
	output = model_fit.forecast() # generate array
	yhat = output[0]
	predictions.append(yhat) 
	obs = test[t] 
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='green')
pyplot.show()
