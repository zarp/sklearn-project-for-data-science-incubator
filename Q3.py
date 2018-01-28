import pandas as pd

import sklearn as sk
from sklearn.cross_validation import train_test_split
from sklearn import pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn.cross_validation import cross_val_predict
from sklearn import grid_search

import os
import dill

def cat_splitter(inp_str):
    """ use counts, not bool!""" 
    keys=inp_str.split(',')
    values=[1]*len(keys) 
    return dict(zip(keys, values))

class q3_main():
    def __init__(self):
        pass
    def train(self):
        filename=os.path.join('data',"yelp_train_academic_dataset_business.csv")
        data = pd.read_csv(filename, low_memory=False)
        input_df=pd.DataFrame({"categories":data['categories'],"stars":data['stars']}) 
        input_df=input_df[input_df.categories.notnull()]
        input_df['categories']=input_df['categories'].apply(cat_splitter)
        training=list(input_df['categories'])
        
        self.categ=DictVectorizer()

        X = self.categ.fit_transform(training)
        Xarr=X.toarray() 

        X=Xarr 
        Y=input_df['stars'] 

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        mylinreg=linear_model.RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0))
        mylinreg.fit(X_train, Y_train)
        self.X_test=X_test
        self.mylinreg=mylinreg
        
        return mylinreg
    def dummy_predict(self): 
        answer=self.mylinreg.predict(self.X_test[0])[0]
        return answer
    def predict(self,inp_record): #
        keys=inp_record['categories']
        values=[1]*len(keys)
        sanitized_entry=dict(zip(keys, values))
        rec_ready_for_linreg=self.categ.transform([sanitized_entry]) 
        answer=self.mylinreg.predict(rec_ready_for_linreg)[0]
        return answer

class ColumnSelectTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):
   def __init__(self):
      pass
   def transform(self, df, y=None, cols=['categories']):
      filename=os.path.join('data',"yelp_train_academic_dataset_business.csv")
      data = pd.read_csv(filename, low_memory=False)
      input_df=pd.DataFrame({"categories":data['categories'],"stars":data['stars']})
      input_df=input_df[input_df.categories.notnull()]
      input_df['categories']=input_df['categories'].apply(cat_splitter)
      training=list(input_df['categories'])
      print('list of cats sent to vectorizer')
      return training
   
   def fit(self):
       return self

class myDictVectorizer(sk.feature_extraction.DictVectorizer):
   def __init__(self): 
     pass
   def transform(training):
      categ=DictVectorizer()
      X = categ.fit_transform(self.training)
      Xarr=X.toarray()
      return Xarr
   def fit(self):
       return self
      
class myRidge(sk.linear_model.RidgeCV):
   def __init__(self): 
      self.Ys=input_df['stars']
      self.Xarr=Xarr
      #pass
   def fit(self,X, Y):
      X=self.Xarr
      Y=iself.Ys
      X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
      mylinreg=linear_model.RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0)) 
      mylinreg.fit(X_train, Y_train)      
      return self
   def predict(self):
      return mylinreg.predict(X_test[0])[0] 

       
q3_ans=q3_main()
q3_ans.train()


q3_pipe=pipeline.Pipeline([
('training', ColumnSelectTransformer()), 
('dictvector', myDictVectorizer()),
('linear', myRidge)  
])


dill.dump(q3_ans, open("q3dill", "wb"))

print("Done!")
