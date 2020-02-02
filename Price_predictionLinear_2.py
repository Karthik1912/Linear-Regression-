import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'https://raw.githubusercontent.com/andradejunior/price_prediction/master/new_properties.csv',index_col='apn')
print(df.head())
print(df.describe())
print(df.columns)

#graphic of the no.of bedrooms
df['beds'].value_counts().plot(kind='bar')
plt.title('Number of Bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
plt.show()

#Grapic price vs bedrooms
plt.scatter(df['price'],df['beds'])
plt.title('Price Vs Berdoome')
plt.xlabel('Price')
plt.ylabel('Bedrooms')
plt.show()


#Grapic price vs bathrooms
plt.scatter(df['price'],df['baths'])
plt.title('Price Vs Bathrooms')
plt.xlabel('Price')
plt.ylabel('Baths')
plt.show()

from sklearn.model_selection import train_test_split

labels = df['price']
train = df.drop(['price','property_type'],axis=1)
X_train,X_test,y_train,y_test = train_test_split(train,labels,test_size=0.10,random_state=2)

from sklearn.linear_model import  LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
print(reg.score(X_test,y_test))

from sklearn import ensemble

#Using Gradient Boosting Regressor classifier and see the accuracy
clf = ensemble.GradientBoostingClassifier(n_estimators=1000,max_depth=15,min_samples_split=2,learning_rate=0.1)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

#Plot the relation beetween test output and Linear Regression model output(expecting linear graphic as ideal)
plt.scatter(reg.predict(X_test), y_test)
plt.show()


#Plot the relation beetween test output and Gradient Boosting Regressor model output(expecting linear graphic as ideal)
plt.scatter(clf.predict(X_test), y_test)
plt.show()

from sklearn.metrics import mean_absolute_error

#Verify the mean absolute percentage error for comparison
mean_absolute_error(y_test, reg.predict(X_test)) #MAPE of Linear regression

mean_absolute_error(y_test, clf.predict(X_test))#MAPE of Gradient Boosting Regressor
