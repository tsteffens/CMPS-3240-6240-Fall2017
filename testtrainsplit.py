

# Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

#reads file into data
df = pd.read_csv('housing_prices.csv')
df['City'] = df['City'].map({'Mountain View': 0, 'Camarillo': 1, 'La Mirada': 2, 'Santa Cruz': 3, 'Santa Clarita': 4, 'Paso Robles': 5, 'Montecito': 6,
                            'Carpinteria': 7, 'Roseville': 8, 'Clairmont': 9, 'Corona': 10, 'Oxnard': 11, 'Tracy': 12,
                            'Chino': 13, 'pleasanton': 14, 'Seal Beach': 15, 'Carmel Valley': 16, 'Moraga': 17,
                            'Redondo Beach': 18, 'Atascadero': 19, 'Antioch': 20, 'Goleta': 21, 'Marina': 22, 'Costa Mesa': 23, 'Tustin': 24, 'modesto': 25,
                            'Danville': 26, 'Lake Forest': 27, 'Saratoga': 28, 'El Dorado Hills': 29, 'Dublin': 30, 'Shingle Springs': 31, 'Cambria': 32,
                            'Oceanside': 33, 'McKinleyville': 34, 'San Luis Obispo': 35, 'Canyon Lake': 36, 'Central San Francisco': 37, 'Rohnert Park City': 38, 'Morro Bay': 39, 'Redwood City' : 40, 'Salida' : 41, 'Rocklin' :42, 'Folsom' : 43, 'Clayton' : 44, 'Elk Grove' : 45, 'Morgan Hill' : 46, 'Benicia': 47})


# Assuming same lines from your example
#cols_to_norm = ['price','sqft','bedrooms','baths','City']
#df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

#splits data into training and testing sets
seed = 7
X_train, X_test, y_train, y_test = train_test_split(df, df['price'],test_size = .3, random_state=seed)



#shape
print("Shape of data Array: ")
print(df.shape)
print(df.describe())


#Univariate plots
df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()

#Multivariate plots
scatter_matrix(df)
plt.show()

#run on train data
neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(X_train, y_train)
KNeighborsClassifier
pred = neigh.predict(X_train)
print(pred)
y_train = y_train.tolist()
print(y_train)
print(accuracy_score(y_train, pred))

#run on test data
neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(X_train, y_train)
KNeighborsClassifier
pred = neigh.predict(X_test)
print(pred)
y_test = y_test.tolist()
print(y_test)
print(accuracy_score(y_test, pred))

