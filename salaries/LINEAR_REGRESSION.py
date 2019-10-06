#################### LINEAR REGRESSION ############

#Importing Libraries
import pandas as pd  
#import numpy as np  
import matplotlib.pyplot as plt  

#imports the CSV dataset using pandas

dataset = pd.read_csv('student_scores.csv')  

#explore the dataset
print (dataset.shape)
print (dataset.ndim)
print (dataset.head())
print (dataset.describe())

#check data types for each column
print (dataset.dtypes)

#Check if any NaN values in dataset
dataset.isnull().any(axis=0)

# Check for outlier values
plt.boxplot(dataset.values)

#let's plot our data points on 2-D graph to eyeball our dataset
# and see if we can manually find any relationship between the data.
dataset.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

#prepare the data to train the model
features = dataset.iloc[:, :-1].values  
labels = dataset.iloc[:, 1].values

"""
train the model now
"""

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(features, labels)

#To see the value of the intercept and slop calculated by the linear regression algorithm for our dataset, execute the following code.
print(regressor.intercept_)  
print (regressor.coef_)

#what would be prediction of score if someone studies 9 hours
print (regressor.coef_*9 + regressor.intercept_)
print (regressor.predict(9))


#Visualize the best fit line
import matplotlib.pyplot as plt

# Visualising the  results
plt.scatter(features, labels, color = 'red')
plt.plot(features, regressor.predict(features), color = 'blue')
plt.title('Study Hours and Exam Score')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score: Marks')
plt.show()
