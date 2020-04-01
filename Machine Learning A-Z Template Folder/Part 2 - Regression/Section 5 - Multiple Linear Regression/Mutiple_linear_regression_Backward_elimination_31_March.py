import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values #will not include the last value also its size is different than y even its 1 colm x and 1 colm y
y = dataset.iloc[:,4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

#Avoiding the Dummy varaible trap
# It means if there are 3 variable 2 can be showen and if in both the value is 0
#then that means it would be in third
X = X[:,1:]

#splitting the dataset into training and test dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)

#fitting the Multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Prediction
y_pred = regressor.predict(X_test)

#------------------------------------------------------------------------
#Building the optimal model with BackwardBackward Elimination 
#------------------------------------------------------------------------
import statsmodels.regression.linear_model as lm

#add a column of 1 to X because x raise to power o = 1 whihc is a constant*xo
#that is INTERCEPT
X = np.append(arr =np.ones([50,1]).astype(int),values = X,axis=1)
X_opt = X[:,[0,1,2,3,4,5]] 
regressor_OLS = lm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#checkthe summary, if  P>|t| is greater than 5 percent(0.05) remove that column

X_opt = X[:,[0,1,2,3,5]] 
regressor_OLS = lm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,2,5]] 
regressor_OLS = lm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()






