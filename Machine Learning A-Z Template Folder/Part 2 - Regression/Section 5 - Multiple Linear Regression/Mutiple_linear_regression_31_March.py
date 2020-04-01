import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values #will not include the last value also its size is different than y even its 1 colm x and 1 colm y
y = dataset.iloc[:,4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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
