import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import datetime 
import numpy as np
import pickle 

list_of_targets = ["System total load in MAW","Wind Offshore in MAW","Wind Onshore in MAW","Solar in MAW"]



def complete_prediction(data,features):
    """this function takes the complete dataset and a list of selected features
    then fit ML models to that data and save those files in a models folder
    it will print the MAPE results of the different models and
    afterwards it will return a new dataset that includes the prediction made by these models"""
    
    X = data.drop(list_of_targets,axis=1)
    y = data.loc[:,list_of_targets]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=None, shuffle=False)
    
    X_train = X_train.loc[:,features]
    X_test = X_test.loc[:,features]
    
    maes = []
#     list_of_models_as_functions = [decision_tree(_,_,_), random_forest(_,_,_)]
    list_of_models = ["Decision Tree","Random Forrest", "Entso-e"]
    
    for i in list_of_targets:
        y_train_specific = y_train.loc[:,i]
        y_test_specific = y_test.loc[:,i]
        
        tree = decision_tree(X_train, y_train_specific,i)
        RF = random_forest(X_train,y_train_specific,i)
        
        maes.append(MAE(y_test_specific,tree.predict(X_test)))
        maes.append(MAE(y_test_specific,RF.predict(X_test)))
        maes.append(MAE(y_test_specific,X.loc[:,('predicted_'+i)]))
    
    
    x = 0
    for i in list_of_targets:
        print_str = ""
        print(("For "+i+" the MAEs are:"))
        for j in range(len(list_of_models)):
            error = (round(maes[j+x*len(list_of_models)],2))
            print_str += f"{list_of_models[j]} = {error}, "
        print(print_str[:-1])
        x+=1
            

    pass
#     return data_including_our_prediction

def decision_tree(X_train,y_train,target):
    model = tree.DecisionTreeRegressor()
    model = model.fit(X_train,y_train)   
    with open(("./models/DecisionTreeModel_"+target+".pickle"),"wb") as f:
        pickle.dump(model, f)
    return model

def random_forest(X_train,y_train,target):
    model = RandomForestRegressor()
    model = model.fit(X_train,y_train)  
    with open(("./models/RandomForestModel_"+target+".pickle"),"wb") as f:
        pickle.dump(model, f)  
    return model


def MAE(y_actual,y_tested):
    return np.mean(np.abs(y_actual-y_tested))