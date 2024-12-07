# Install libraries not already in the environment using pip
#!pip install pandas==1.3.4
#!pip install sklearn==0.20.1


# Pandas will allow us to create a dataframe of the data so it can be used and manipulated
import pandas as pd
# Regression Tree Algorithm
from sklearn.tree import DecisionTreeRegressor
# Split our data into a training and testing data
from sklearn.model_selection import train_test_split

data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/real_estate_data.csv")

data.head()

data.isna().sum()

data.dropna(inplace = True)

data.isna().sum()


# Lets split the data into our features and predicting(target)

X = data.drop(columns=["MEDV"])
Y = data["MEDV"]

# Now lets split our data into a training and testing dataset using `train_test_split` from `sklearn.model_selection`
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=1)

regression_tree = DecisionTreeRegressor(criterion='friedman_mse')
regression_tree.fit(X_train, Y_train)

print(regression_tree.score(X_test, Y_test))

prediction = regression_tree.predict(X_test)

print("$",(prediction - Y_test).abs().mean()*1000)