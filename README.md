# task_1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d="http://bit.ly/w-data"
data=pd.read_csv(d)
print("Data imported successfully")

data.head()

data.tail()

data.columns

data.info

data.describe()

data.plot(x="Hours",y="Scores",style=".")
plt.title("hrs vs Scores")
plt.xlabel('hrs')
plt.ylabel("score %")
plt.show()

data.plot(kind='bar',figsize=(10,5))
plt.title("hrs bs score %")
plt.xlabel("steady hrs")
plt.show()

x=data.iloc[:,:-1].values
y=data.iloc[:, 1].values

from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                            test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(x_train, y_train) 

print("Training is completed")

line = regressor.coef_*x+regressor.intercept_
plt.scatter(x, y)
plt.plot(x, line);
plt.show()

print(x_test)
y_pred=regressor.predict(x_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df

hrs=9.25
pred=regressor.predict(np.array(hrs).reshape(-1,1))
print("no of hrs={}".format(pred[0]))
print("predicted score={}".format(pred[0]))

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 
print('Root mean square erroe:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("Mean squared error:",metrics.mean_squared_error(y_test,y_pred))
