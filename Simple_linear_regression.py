import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

df = pd.read_csv(r'C:\Users\kartheek\Downloads\ML\100-Days-Of-ML-Code-master\datasets/studentscores.csv')

X= df.iloc[:,:1].values
Y = df.iloc[:,1].values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=.20,random_state=0)

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression = regression.fit(X_train,y_train)

y_pred = regression.predict(X_test)

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regression.predict(X_train),color='green')

plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regression.predict(X_test),color='green')
plt.show()
