# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #we added 2 , but two is upper bound , becase we want it as matrics not vector
y = dataset.iloc[:, 2].values

# we are not spitting and testing because we need acccurate and specific data and data is very less
#so we just simply use polynomial regression

#fitting linear regression
#making it linear just to compare it with poly.. otherwise no need
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y) 

#first we transform into polynomial fit AND THEN  we fit that in linear model
from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree = 4)# at first i took 2,3 degree but 4 is giving better result
X_poly = polyreg.fit_transform(X)
#now fit
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

#visualization of Linear rgression results
plt.scatter(X,y,color ='red')
plt.plot(X,linreg.predict(X),color='blue')
plt.title('Truth or bluf linear')

#visualization of poly rgression results
plt.scatter(X,y,color ='red')
plt.plot(X,lin_reg.predict(polyreg.fit_transform(X)),color='blue')
#using polyreg.fit and not X_poly because we can later just subsitute x and into x_grid ands not forget totransform
plt.title('Truth or bluf poly')

#to get more continues curve we can 
x_grid = np.arange(min(X),max(X),0.1 )
x_grid =  x_grid.reshape(len(x_grid),1)
#plot again to see detailed graph
plt.scatter(X,y,color ='red')
plt.plot(x_grid,lin_reg.predict(polyreg.fit_transform(x_grid)),color='blue')
plt.title('Truth or bluf poly')


#check if the empoye said with 2 years after 6  level(from 6-7 4 years) if his salary is around 160000 or not
var = [6.5]
var = np.asarray(var)
var =  var.reshape(1,1)
print(lin_reg.predict(polyreg.fit_transform(var)))

#hence accurate 









