# Stock Prediction
Predict Stock Price (Adj Close) using a variety of Machine Learning Algorithms

AAPL equity Close Price is trained using:
1. Linear Regression
1. Quadratic Regression - polynomials of order 2 and 3
1. KNN Regression
1. Lasso Regression - No of features 2 (alpha = none, alpha = 0.01, alpha = 0.0001

## Training and Test Scores
### Linear Regression
inear Regression training score: 0.967404228040554
Linear Regression test score:  0.9673076257330185

### Quadratic Poly2 Regressio
Quadrartic Poly2 Regression training score: 0.9688253558181589
Quadratrci Poly2  test score:  0.9689579969047571

### Quadratic Poly3 Regression
Quadrartic Poly3 Regression training score: 0.9705560643936337
Quadratrci Poly3  test score:  0.9705514356135095

### KNN
KNN training score: 0.9882769020929683
KNN test score:  0.9580742423824369

### Lasso Regression, Alpha = 0, Feature = 2
Lasso Regression training score: 0.9669421078029856
Lasso Regression test score:  0.967465574465809
Lasso Regression number of features used:  2

### Lasso Regression, Alpha = 0.01, Feature = 2
Lasso Regression training score for alpha=0.01: 0.9674041874211028
Lasso Regresion test score for alpha =0.01:  0.9673126700132454
Lasso Regression number of features used: for alpha =0.01: 2

### Lasso Regression, Alpha = 0.0001, Feature = 2
Lasso Regression training score for alpha=0.0001: 0.9674042280340734
Lasso Regression test score for alpha =0.0001:  0.9673076035316527
Lasso Regression number of features used: for alpha =0.0001: 2

# Explanation of the Prediction graph
The actual data is plotted using Red Color. For the data set used,
Linear Regression seems to predict the result.

[AAP Equity Prediction](../)

