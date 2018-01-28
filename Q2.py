import pandas as pd
import sklearn as sk
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import grid_search
from sklearn import pipeline
import os
import dill

class ColumnSelectTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):

   def __init__(self):
       pass

   def transform(self, data, y=None, cols=['latitude', 'longitude']):
       return data[cols]

   def fit(self, data, y=None):
       return self
      
filename=os.path.join('data',"yelp_train_academic_dataset_business.csv")
data=pd.DataFrame.from_csv(filename)
training = data[['latitude','longitude']]

X_train, X_test, y_train, y_test = train_test_split(training, data.stars, test_size=0.3, random_state=42)

parameters = {'n_neighbors' : [1,5,10,15,20]}
neigh =  KNeighborsRegressor()
modelneigh = grid_search.GridSearchCV(neigh, parameters) 

modelneigh.fit(X_train,y_train)
pred = modelneigh.predict(X_test)
results = modelneigh.score(X_test, y_test)

q2_pipe = pipeline.Pipeline([
  ('column_select', ColumnSelectTransformer()),
  ('modelneigh', grid_search.GridSearchCV(neigh, parameters))
  ])
  
q2_pipe.fit(X_train, y_train)
dill.dump(modelneigh, open("q2dill", "wb"))

print("DONE!")
