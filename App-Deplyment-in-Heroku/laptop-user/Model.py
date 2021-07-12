import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

agent = pd.read_csv('Laptop-Users')

X = agent[['Avg. Area Age','Avg. Area Gender','Avg. Area Region','Avg. Area Occupation','Avg. Area Income']]
y = agent['has-laptop']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)

lm = LinearRegression()
lm.fit(X_train,y_train)

pickle.dump(lm, open('model.pkl', 'wb'))
