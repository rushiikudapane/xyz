import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("Employee_Salary_Dataset.csv")
print(df)
# print(df.describe())
# columns = ["Experience_Years", "Age", "Salary"]
# for column in columns:
#     m1 = df[column].min()
#     m2 = df[column].max()
#     m3 = df[column].mean()
#     m4 = df[column].median()
#     m5 = df[column].std() 
#     # print("Column_Name :",column)
#     # print("min :",m1)
#     # print("max :",m2)
#     # print("mean :",m3)
#     # print("median :",m4)
#     # print("std :",m5)

# # # group by gender
# columnName = ["Experience_Years", "Age", "Salary"]
# for column in columnName:
#     # print("++++++++++++++++++++",column,"++++++++++++++++++++")
#     # print("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}".format('Column_Name :','min','max','mean','median','std'))
#     m1 = df[column].groupby(df['Gender']).min()
#     m2 = df[column].groupby(df['Gender']).max()
#     m3 = df[column].groupby(df['Gender']).mean()
#     m4 = df[column].groupby(df['Gender']).median()
#     m5 = df[column].groupby(df['Gender']).std()
#     # print("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}".format('Female',m1[0],m2[0],m3[0],m4[0],m5[0]))
#     # print("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}".format('Male',m1[1],m2[1],m3[1],m4[1],m5[1]))

# df1 = pd.DataFrame(columns = ["Experience_Years", "Age", "Salary"])
# X=['min','max','mean','median','std']
# x_axis = np.arange(len(X))

# columnName = ["Experience_Years", "Age", "Salary"]
# for column in columnName:
#     n1 = df1[column].groupby(df['Gender']).min()
#     n2 = df1[column].groupby(df['Gender']).max()
#     n3 = df1[column].groupby(df['Gender']).mean()
#     n4 = df1[column].groupby(df['Gender']).median()
#     n5 = df1[column].groupby(df['Gender']).std()
#     Y=[n1,n2,n3,n4,n5]
    

# for i in columns:
# 	if i=='Experience_Years':
# 		s1=df.groupby(['Gender'])[i].min()
# 		s2=df.groupby(['Gender'])[i].max()
# 		s3=df.groupby(['Gender'])[i].mean()
# 		s4=df.groupby(['Gender'])[i].std()

# 		dict= {
# 				"sex":['Female','Female','Female','Female','Male','Male','Male','Male'],
# 				"Experience_Years":['min','max','mean','std','min','max','mean','std'],
# 				"value":[s1[0],s2[0],s3[0],s4[0],s1[1],s2[1],s3[1],s4[1]]
# 		}

# 		#print("-------------",i,"-------------")
# 		df1=pd.DataFrame(dict)
# 		#print(df1)

# 	elif i=='Age':
# 		s1=df.groupby(['Gender'])[i].min()
# 		s2=df.groupby(['Gender'])[i].max()
# 		s3=df.groupby(['Gender'])[i].mean()
# 		s4=df.groupby(['Gender'])[i].std()

# 		dict= {
# 				"sex":['Female','Female','Female','Female','Male','Male','Male','Male'],
# 				"Age":['min','max','mean','std','min','max','mean','std'],
# 				"value":[s1[0],s2[0],s3[0],s4[0],s1[1],s2[1],s3[1],s4[1]]
# 		}

# 		#print("-------------",i,"-------------")
# 		df2=pd.DataFrame(dict)
# 		#print(df2)

# 	else:
# 		s1=df.groupby(['Gender'])[i].min()
# 		s2=df.groupby(['Gender'])[i].max()
# 		s3=df.groupby(['Gender'])[i].mean()
# 		s4=df.groupby(['Gender'])[i].std()

# 		dict= {
# 				"sex":['Female','Female','Female','Female','Male','Male','Male','Male'],
# 				"Salary":['min','max','mean','std','min','max','mean','std'],
# 				"value":[s1[0],s2[0],s3[0],s4[0],s1[1],s2[1],s3[1],s4[1]]
# 		}

# 		#print("-------------",i,"-------------")
# 		df3=pd.DataFrame(dict)
# 		# print(df3)

# sns.barplot(data=df1, x="Experience_Years", y="value", hue="sex")
# plt.show()
# sns.barplot(data=df2, x="Age", y="value", hue="sex")
# plt.show()
# sns.barplot(data=df3, x="Salary", y="value", hue="sex")
# plt.show()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# df = pd.read_csv("./Iris.csv")
# print(df)
# print(df.describe())
# columns = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
# for column in columns:
# 	m1 = df[column].min()
# 	m2 = df[column].max()
# 	m3 = df[column].mean()
# 	m4 = df[column].median()
# 	m5 = df[column].std()
# 	print("column Name :",column)
# 	print("min : ",m1)
# 	print("max :",m2)
# 	print("mean :",m3)
# 	print("median :",m4)
# 	print("std :",m5)
# # group by species 
# for column in columns:
	# print("++++++++++++++++++",column,"+++++++++++++++++")
	# print("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}".format('Column_Name :','min','max','mean','meadian','std'))
	# m1 = df[column].groupby(df['Species']).min()
	# m2 = df[column].groupby(df['Species']).max()
	# m3 = df[column].groupby(df['Species']).mean()
	# m4 = df[column].groupby(df['Species']).median()
	# m5 = df[column].groupby(df['Species']).std()
	# print("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}".format('Iris-setosa',m1[0],m2[0],m3[0],m4[0],m5[0]))
	# print("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}".format('Iris-virginica',m1[1],m2[1],m3[1],m4[1],m5[1]))
   


