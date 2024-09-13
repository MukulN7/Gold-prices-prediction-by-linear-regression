#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

#defining variables and putting approriate values in them for further use
df = pd.read_csv('path to read gold prices & relevant metrics dataset in csv form')
p2dp = df['Price 2 Days Prior']
p1dp = df['Price 1 Day Prior']
p_today = df['Price Today']
X = np.column_stack((p2dp.values,p1dp.values))
Y = p_today.values

#Making the linear regression model
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
model = LinearRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

#Evaluating the model
print(Y_pred)
mse = mean_squared_error(Y_test,Y_pred)
r2 = r2_score(Y_test,Y_pred)
print(f"Mean squared error is: {mse}\nR2 Score = {r2}")

#Code for visualisation of process

#Plotting the dataset
fig = plt.figure(figsize=(12,7),edgecolor='Brown',facecolor='Pink')
ax1 = fig.add_subplot(122,projection='3d',facecolor='pink')
ax1.scatter(X_train[:,0],X_train[:,1],Y_train,c='Red',s=7,marker='o',label='Training Data')
ax1.set_title('Gold Price Prediction')
ax1.set_xlabel('Price 2 days prior')
ax1.set_ylabel('Price 1 day prior')
ax1.set_zlabel('Price today')
ax1.scatter(X_test[:,0],X_test[:,1],Y_test,c='Blue',s=22,marker='x',label='Testing Data')
#Plotting the linear regression model
x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 10)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
Z_grid = model.predict(np.c_[X1_grid.ravel(), X2_grid.ravel()])
Z_grid = Z_grid.reshape(X1_grid.shape)
ax1.plot_surface(X1_grid, X2_grid, Z_grid, color='Yellow', alpha=0.3, label='Regression Plane')
ax1.legend(loc='upper right')
#plotting the dataset again on the left without the linear regression model
ax2 = fig.add_subplot(121,projection='3d',facecolor='Pink')
ax2.scatter(X_train[:,0],X_train[:,1],Y_train,c='Red',s=7,marker='o',label='Training Data')
ax2.set_title('Gold Price Dataset')
ax2.set_xlabel('Price 2 days prior')
ax2.set_ylabel('Price 1 day prior')
ax2.set_zlabel('Price today')
ax2.scatter(X_test[:,0],X_test[:,1],Y_test,c='Blue',s=22,marker='x',label='Testing Data')
ax2.legend(loc='upper right')
plt.show()