import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataframe = pd.read_csv(r"C:\Users\kartheek\Downloads\Housing_Price_Analysis/Melbourne_housing_extra_data.csv")
print(dataframe.info())
print(dataframe.columns)
print(dataframe.describe())
print(dataframe.head())

dataframe['Date'] = pd.to_datetime(dataframe['Date'],dayfirst=True) # Convert Date
print(len(dataframe['Date'].unique())/4)#12 means a year of data

var = dataframe[dataframe['Type']=='h'].sort_values('Date',ascending=False).groupby('Date').std()
count= dataframe[dataframe['Type']=='h'].sort_values('Date',ascending=False).groupby('Date').count()
mean = dataframe[dataframe['Type']=='h'].sort_values('Date',ascending=False).groupby('Date').mean()

mean['Price'].plot(yerr = var['Price'],ylim=(400000,1500000))
#plt.show()

means = dataframe[(dataframe["Type"]=="h") & (dataframe["Distance"]<13)].sort_values("Date", ascending=False).groupby("Date").mean()
errors = dataframe[(dataframe["Type"]=="h") & (dataframe["Distance"]<13)].sort_values("Date", ascending=False).groupby("Date").std()
print(means.columns)

means.drop(["Price",
            "Postcode",

           "Longtitude","Lattitude",
           "Distance","BuildingArea", "Propertycount","Landsize","YearBuilt"],axis=1).plot(yerr=errors)
#plt.show()

dataframe[dataframe['Type']=='h'].sort_values('Date',ascending=False).groupby('Date').mean()

pd.set_eng_float_format(accuracy=1,use_eng_prefix=True)
dataframe[(dataframe['Type']=='h') &
          (dataframe['Distance']<14) &
          (dataframe['Distance']>13.7)
          #&(dataframe['Suburb'] =='Northcote')
 ].sort_values('Date',ascending=False).dropna().groupby(['Suburb','SellerG']).mean()

#print(dataframe)
dataframe.dropna()
sns.kdeplot(dataframe[(dataframe['Suburb']=='Northcote')&
                      (dataframe['Type']=='u')&
                      (dataframe['Rooms']==2)]['Price'])

plt.show()