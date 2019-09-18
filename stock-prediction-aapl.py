import pandas as pd
import numpy as np
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import math
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

start = datetime.datetime(2010, 1, 1)
end = datetime.date.today()

# read AAPL quotes from 1/1/2017 to 1/1/2019
df = web.DataReader("AAPL", 'yahoo', start, end)



# Feature Engineering
dfreg = df.loc[:,["Adj Close","Volume"]]
df['HL_PCT'] = (df['High']-df['Low'])/df['Close'] * 100.0
df['PCT_change'] = (df['Close']-df['Open'])/df['Open'] * 100.0

# Drop Missing value
dfreg.fillna(value=-99999, inplace=True)

# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))
#print(len(dfreg))
#print(forecast_out)

# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = Xadj = np.array(dfreg.drop(['label'], 1))

# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)

# Finally We want to find Data Series of late X and early X (train) 
# for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
Xact1 = Xadj[-forecast_out:]
Xact = Xact1[:,0]




# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)
train_score=clfreg.score(X_train,y_train)
test_score=clfreg.score(X_test,y_test)
print("Linear Regression training score:", train_score)
print("Linear Regression test score: ", test_score)
forecast_reg = clfreg.predict(X_lately)

#xl = X_lately['Adj Close'].iloc[:6,:1]
#print(xl.shape)


# Compute Date - Y axis
xaxis = []
last_date = dfreg.iloc[-1].name
last_date1 = last_date
next_date1 = last_date1 + datetime.timedelta(days=1)
for i in forecast_reg:
        next_date = next_date1
        xaxis.append(next_date)
        next_date1 += datetime.timedelta(days=1)



# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)
train_score=clfpoly2.score(X_train,y_train)
test_score=clfpoly2.score(X_test,y_test)
print("Quadrartic Poly2 Regression training score:", train_score)
print("Quadratrci Poly2  test score: ", test_score)
forecast_poly2 = clfpoly2.predict(X_lately)


# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)
train_score=clfpoly3.score(X_train,y_train)
test_score=clfpoly3.score(X_test,y_test)
print("Quadrartic Poly3 Regression training score:", train_score)
print("Quadratrci Poly3  test score: ", test_score)
forecast_poly3 = clfpoly3.predict(X_lately)




# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)
train_score=clfknn.score(X_train,y_train)
test_score=clfknn.score(X_test,y_test)
print("KNN training score:", train_score)
print("KNN test score: ", test_score)
forecast_knn = clfknn.predict(X_lately)


# Lasso Regression
cllasso = Lasso()
cllasso.fit(X_train, y_train)
train_score=cllasso.score(X_train,y_train)
test_score=cllasso.score(X_test,y_test)
coeff_used = np.sum(cllasso.coef_!=0)
forecast_cllasso = cllasso.predict(X_lately)

print("Lasso Regression training score:", train_score)
print("Lasso Regression test score: ", test_score)
print("Lasso Regression number of features used: ", coeff_used)


cllasso001 = Lasso(alpha=0.01, max_iter=10e5)
cllasso001.fit(X_train,y_train)
train_score001=cllasso001.score(X_train,y_train)
test_score001=cllasso001.score(X_test,y_test)
coeff_used001 = np.sum(cllasso001.coef_!=0)
forecast_cllasso001 = cllasso001.predict(X_lately)

print("Lasso Regression training score for alpha=0.01:", train_score001)
print("Lasso Regresion test score for alpha =0.01: ", test_score001)
print("Lasso Regression number of features used: for alpha =0.01:", coeff_used001)



cllasso00001 = Lasso(alpha=0.0001, max_iter=10e5)
cllasso00001.fit(X_train,y_train)
train_score00001=cllasso00001.score(X_train,y_train)
test_score00001=cllasso00001.score(X_test,y_test)
coeff_used00001 = np.sum(cllasso00001.coef_!=0)
forecast_cllasso00001 = cllasso00001.predict(X_lately)

print("Lasso Regression training score for alpha=0.0001:", train_score00001)
print("Lasso Regression test score for alpha =0.0001: ", test_score00001)
print("Lasso Regression number of features used: for alpha =0.0001:", coeff_used00001)


# Plot all the Prediction Algorithm along with the actual
# Adjusting the style of matplotlib
style.use('ggplot')
plt.xlabel('Date')
plt.ylabel('Prediction')
plt.plot(xaxis, Xact, color='Red')
plt.plot(xaxis, forecast_reg)
plt.plot(xaxis, forecast_poly2)
plt.plot(xaxis, forecast_poly3)
plt.plot(xaxis, forecast_knn)
plt.plot(xaxis, forecast_cllasso)
plt.plot(xaxis, forecast_cllasso001)
plt.plot(xaxis, forecast_cllasso00001)
plt.legend()
plt.show()