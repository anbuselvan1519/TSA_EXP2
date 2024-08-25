## Developed by: Anbuselvan S
## Register No: 212223240008
## Date: 

# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION

### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
     Import necessary libraries (NumPy, Matplotlib)
     Load the dataset
     Calculate the linear trend values using least square method
     Calculate the polynomial trend values using least square method
     End the program

### PROGRAM:
#### A - LINEAR TREND ESTIMATION:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

data = pd.read_csv("/content/yahoo_stock.csv")

data['Date'] = pd.to_datetime(data['Date'])
data['Date_ordinal'] = data['Date'].map(pd.Timestamp.toordinal)


x = data[['Date_ordinal']].values
y = data['Volume'].values

linear_model = LinearRegression()
linear_model.fit(x, y)
y_linear_pred = linear_model.predict(x)
m = linear_model.coef_[0]
c = linear_model.intercept_

plt.plot(x, y,color='black', label='Data Points')

plt.plot(x, y_linear_pred, color='red', label='Linear Trend')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data and Trends')
plt.legend()
plt.show()
```
#### B- POLYNOMIAL TREND ESTIMATION:
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

data = pd.read_csv("/content/yahoo_stock.csv")

data['Date'] = pd.to_datetime(data['Date'])
data['Date_ordinal'] = data['Date'].map(pd.Timestamp.toordinal)

x = data[['Date_ordinal']].values
y = data['Volume'].values

degree = 2
poly_features = PolynomialFeatures(degree)
x_poly = poly_features.fit_transform(x)
poly_model = LinearRegression()
poly_model.fit(x_poly, y)
y_poly_pred = poly_model.predict(x_poly)

plt.plot(x, y,color='black', label='Data Points')

x_fit = np.linspace(min(x), max(x), 100).reshape(-1, 1)
x_fit_poly = poly_features.transform(x_fit)
y_fit_poly = poly_model.predict(x_fit_poly)
plt.plot(x_fit, y_fit_poly, color='yellow', label='Polynomial Trend')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data and Trends')
plt.legend()
plt.show()
```

### OUTPUT:
#### A - LINEAR TREND ESTIMATION:
![image](https://github.com/user-attachments/assets/c396c8ec-8b60-442b-8b2a-babb33bcaace)

#### B- POLYNOMIAL TREND ESTIMATION:
![image](https://github.com/user-attachments/assets/6c810146-b0e1-4741-94b9-58224fc8251e)

### RESULT:
Thus, the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
