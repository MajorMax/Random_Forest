    import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Fitting Random Forest Tree Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 3000, random_state = 0)
regressor.fit(X, y)


#Prediction
y_pred = regressor.predict([[6.5]])


#Visualising the RFR Regression Results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position Label')
plt.show()