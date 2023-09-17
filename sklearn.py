# Steps
## Replace
### Additional information

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import openpyxl

# 1) import dataset
data=pd.read_csv(##'F:/Machin Learning/trainapple.csv')
### data.head
      
# 2) inputs and outputs
x=data[ ## column label for example ['Open', 'High', 'Low']]
y=data[ ## column label for example 'Close']
    
# 3) Split dataset into training set and test set
trainX, testX, trainY, testY = train_test_split(x, y, test_size = ## 0.2)
                                                
# 4) Scaling                                               
sc=StandardScaler()

scaler = sc.fit(trainX)
trainX_scaled = scaler.transform(trainX)
testX_scaled = scaler.transform(testX)
                                                
# 5) Hyper parameters
mlp_reg = MLPRegressor( ## hidden_layer_sizes=(500,180,50), max_iter = 300, activation = 'relu', solver = 'adam', random_state=5, learning_rate_init=0.01 )
                                                
# 6) train network
mlp_reg.fit(trainX_scaled, trainY)
                                
# 7) test network 
y_pred = mlp_reg.predict(testX_scaled)
                                                
# 8) Display forecast and actual data
df_temp = pd.DataFrame({'Actual': testY, 'Predicted': y_pred})
### df_temp.head()
### df_temp.to_excel( ## r'F:/Machin Learning/App2.xlsx')
                                                
# 9) performance                                               
print('Mean Absolute Error:', metrics.mean_absolute_error(testY, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(testY, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(testY, y_pred)))

# 10) loss_curve
plt.plot(mlp_reg.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()
