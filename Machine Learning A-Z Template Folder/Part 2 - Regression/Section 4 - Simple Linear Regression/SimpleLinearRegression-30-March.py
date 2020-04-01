import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values #will not include the last value
y = dataset.iloc[:,1].values

#splitting the dataset into training and test dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

#fitting the simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the result
y_pred = regressor.predict(X_test)
slope = regressor.coef_

#visualise the Training set
plt.scatter(X_train,y_train, color = 'red')
plt.scatter(X_train,regressor.predict(X_train), color ='green')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#Visualize the test data
#we didnt changed the x_train in second line because we trained
#the regressor which has a equation, now it will give same 
#value just the point position might be different
plt.scatter(X_test,y_test, color = 'red')
plt.scatter(X_train,regressor.predict(X_train), color ='green')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()