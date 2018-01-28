import pandas as pd
import os
import sys
import dill

class mypredictor():
    def __init__(self, inp_datafile_name=os.path.join("data","q1_inp_for_model.csv")):
        self.data=pd.DataFrame.from_csv(inp_datafile_name)
        self.average=self.data['stars'].mean()

    def predict(self, input_dict):
        input_record=input_dict
        input_city=input_record['city']
        if input_city in self.data.index:
            return self.data.loc[input_city][0]
        else:
            return self.average

filename=os.path.join("data","yelp_train_academic_dataset_business.csv")
data=pd.DataFrame.from_csv(filename)


q1_data=pd.DataFrame({"city":data['city'],"stars":data['stars']})
q1_data.city[q1_data.city=='chandler'] = 'Chandler' # dealing with inconsistent capitalization for one of the cities
q1_data=q1_data.sort_values(by="city",ascending=True)

q1_ans=q1_data.groupby(['city']).mean()
q1_ans=q1_ans.reset_index()

q1_ans.to_csv(os.path.join("data","q1_inp_for_model.csv"), encoding='utf-8', index=False)  

dill.dump(mypredictor, open(os.path.join("data","q1dill"), "wb"))

print("Done!")
