import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test_df = pd.read_csv('test.csv')
train_df = pd.read_csv('train.csv')

#description = [train_df[x].describe() for x in train_df.columns]
#description

X = train_df.iloc[:,[4,76,77]].values
y = train_df.iloc[:, -1].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
# =============================================================================
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)
# =============================================================================


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
#regressor.score(X_train, y_pred)


# Visualising the Training set results
plt.scatter(X_train[:,0], y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Sales vs LotArea (Training set)')
plt.xlabel('LotArea')
plt.ylabel('Sales Price')
plt.show()

# Visualising the Test set results
plt.scatter(X_test[:,0], y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Sales vs LotArea (Test set)')
plt.xlabel('LotArea')
plt.ylabel('Sales Price')
plt.show()