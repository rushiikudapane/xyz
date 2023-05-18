import pandas as pd
import numpy as np

df=pd.read_csv('Placement_Data_Full_Class.csv')
# print(df)
# print(df.info)
# print(df.shape)
# print(df.columns)
# print(df.size)
# print(df.dtypes)
# print(df.head())
# print(df.tail())
# print(df.sample(5))
# print(df.describe())
# print(df.isna().sum())
# print(df.isnull().sum())
# print(df.info())

df['sl_no']=df['sl_no'].astype('int8')
# print(df.dtypes)
df['ssc_p']=df['ssc_p'].astype('int8')
# print(df.dtypes)
# print(df.dtypes['ssc_p'])

di = { 'Roll':[1,2,3,4,5],
      'Name':['Vivek', 'Aboli', 'Shrikant', 'Sita', 'Vijay'],
      'Marks':['First', 'Distinction', 'Distinction', 'Second', 'First']
}
df1 = pd.DataFrame(di)
# print(df1)

df1['Marks'].replace(['Distinction','First','Second'],[0,1,2],inplace=True)
# print(df1)

df1['Marks']=df1['Marks'].astype('category')
# print('data types of Marks=')
# print(df1.dtypes['Marks'])
df1['Marks']=df1['Marks'].cat.codes
# print('data types of Marks=')
# print(df1.dtypes['Marks'])
# print(df1)

# load data one more time#
# df=pd.read_csv('Placement_Data_Full_Class.csv')
# print(df)

from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
df[["gender"]] = oe.fit_transform(df[["gender"]])
print(df)

df=pd.get_dummies(df,columns=['gender'],prefix='sex').head(100)
print(df)

for i in df.columns:
	df[i].fillna(df[i].mode()[0],inplace=True)
print(df)

df['salary'].replace(['numpy.nan'],df['salary'].mean(),inplace=True)
print(df)


print(df.interpolate().count())
print(df.interpolate())

df2=df['salary'].dropna(axis=0)
print(df2)
print("Old data frame length:", len(df['salary']), "\nNew data frame length:",len(df2), "\nNumber of rows with at least 1 NA value: ",(len(df)-len(df2)))

print('Data normalization')
df['salary']=df['salary']/df['salary'].abs().max()
print(df)

# from sklearn.preprocessing import MaxAbsScaler
# abs_scaler=MaxAbsScaler()
# df['salary']=MaxAbsScaler().fit_transform(df['salary'])
# print('\n Maximum absolute Scaling method normalization -1 to 1 \n\n')
# print(df['salary'])


# df=df/df.abs().max()
# print(df['salary'])
from sklearn.preprocessing import MaxAbsScaler
# abs_scaler=MaxAbsScaler()
# df=MaxAbsScaler().fit(df)
# print(df)

# from sklearn.impute import MissingIndicator
# df['salary'].replace({999.0 : np.NaN}, inplace=True)
# indicator = MissingIndicator(missing_values=np.NaN)
# indicator = indicator.fit_transform(df['salary'])
# indicator = pd.DataFrame(indicator, columns=['salary'])
# print(indicator)


# df['salary']=(df['salary']-df['salary'].min())/(df['salary'].max()-df['salary'].min())
# # print(df)
# from sklearn.preprocessing import MinMaxScaler
# scaler=MinMaxScaler()
# df['salary']=scaler.fit_transform(df['salary'])
# print('\n MinMax feature Scaling method normalization 0 to 1 \n\n')
# print(df['salary'])

# from sklearn.preprocessing import StandardScaler
# df['salary']=(df['salary']-df['salary'].mean())/(df['salary'].std())
# print('\n z score is \n\n')
# print(df['salary'])

# from sklearn.preprocessing import RobustScaler
# df['salary']=(df['salary']-df['salary'].mean())/(df['salary'].quantile(0.75)-df['salary'].quantile(0.25))
# print('\n Robust Scaling \n\n')
# print(df['salary'])

var=max(df["raisedhands"])
print(var)
print(df["raisedhands"])
# MaxAbsScaler(scales such as the maximum value becomes 1 and others scale accordingly)
# method1
var = df["raisedhands"] / df["raisedhands"].abs().max()
print(var)