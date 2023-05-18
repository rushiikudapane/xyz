import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import load_dataset

df = pd.read_csv('titanic.csv' )
print(df)

sns.countplot(df['Survived'])
plt.show()

df['Sex'].value_counts().plot(kind='pie', autopct="%.2f")
plt.show()

plt.hist(df['Age'], bins=5)
plt.show()

sns.distplot(df['Age'])
plt.show()

sns.scatterplot(data=df, x='Fare', y='Age')
plt.show()

sns.scatterplot(df['Fare'], df['Age'], hue=df["Sex"])
plt.show()

sns.scatterplot(df['Fare'], df['Age'], hue=df["Sex"], style=df['Survived'])
# plt.show()

sns.barplot(df['Pclass'], df['Age'])
plt.show()

sns.barplot(df['Pclass'], df['Age'], hue=df['Sex'])
plt.show()

sns.boxplot(df['Sex'], df['Age'])
plt.show()

sns.boxplot(df['Sex'], df['Age'], df["Survived"])
plt.show()

sns.distplot(df[df['Survived'] == 0]['Age'], hist=False, color="blue")
sns.distplot(df[df['Survived'] == 1]['Age'], hist=False, color="orange")
plt.show()


a=pd.crosstab(df['Pclass'], df['Survived'])
print(a)
pd.crosstab(df['Pclass'], df['Survived'])
sns.heatmap(pd.crosstab(df['Pclass'], df['Survived']))
plt.show()

sns.clustermap(pd.crosstab(df['Parch'], df['Survived']))
plt.show()