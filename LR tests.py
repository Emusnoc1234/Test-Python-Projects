#linear regression model tests

from pydataset import data
import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
# matplotlib notebook 
pima = data('Pima.tr')
pima.plot(kind='scatter', x='skin', y='bmi')
plt.show()

# test train split for supervised learning
X_train, X_test, y_train, y_test = train_test_split(pima.skin, pima.bmi)
# test train split visualization
plt.scatter(X_train, y_train, label='Training Data', color='r', alpha=.7)
plt.scatter(X_test, y_test, label='Testing Data', color='b', alpha=.7)
plt.legend()
plt.title("Test Train Split")
plt.show()
#create linear model and train it
LR = LinearRegression()
LR.fit(X_train.values.reshape(-1,1), y_train.values)
#use model to predict on TEST data
prediction = LR.predict(X_test.values.reshape(-1,1))
#plot prediction line against actual test data
plt.plot(X_test, prediction, label='Linear Regression', color='g')
plt.scatter(X_test, y_test, label='Actual Test Data', color='g', alpha=.7)
plt.legend()
plt.show()

#predict BMI of woman with skinfold 50
LR.predict(np.array([[50]]))[0]
#score this model
LR.score(X_test.values.reshape(-1,1), y_test.values)