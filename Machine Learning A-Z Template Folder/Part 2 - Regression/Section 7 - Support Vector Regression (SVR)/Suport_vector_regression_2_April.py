# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#Not all regrssor model have feature scaling and  this class doesnt seem to aply feature scaling so use it
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
#getting shape error so chaning 
y =  y.reshape(-1,1)
y = sc_y.fit_transform(y)

# Fitting the SVR Model to the dataset
# Create your regressor here
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
#we need to feature scale the 6.5
var = np.asarray(6)
var =  var.reshape(-1,1) 
#or  var = np.asarray([[6.5]]) now its shave 1,1
#y_pred = regressor.predict(sc_X.fit_transform(var)) #scaled answer
#to get accurate that is in terms of salary , we need to inverse 
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.fit_transform(var)))



# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

