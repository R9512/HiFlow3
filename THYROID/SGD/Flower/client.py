import warnings
import flwr as fl
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
from random import randint as rand
import sys,toml
global MALICIOUS
global PROP

def mess_the_weights(coef,intercept):
    """
        - Changes the weights and intecept
        - Follows hybrid appraoch, like adding noise and change flipping
        - First adding the noise then flipping the changes
    """
    global PROP

    #Adding Noise
    noise_coef = np.random.normal(loc=0, scale=PROP, size=coef.shape)
    coef+=noise_coef
    noise_intercept = np.random.normal(loc=0, scale=PROP, size=intercept.shape)
    intercept+=noise_intercept
    #Signing Flipping
    n = coef.size
    k = int(PROP*n)
    coef_mask = np.random.choice(n,size=k,replace=False)
    coef_mask = np.unravel_index(coef_mask,coef.shape)
    coef[coef_mask]*=-1
    n = intercept.size
    k = int(PROP*n)
    #print("BEFORE:",intercept)
    intercept_mask = np.random.choice(n,size=k,replace=False)
    intercept_mask = np.unravel_index(intercept_mask,intercept.shape)
    intercept[intercept_mask]*=-1
    # print("AFTER:",intercept)
    # print("="*6)
    print("MODEL WEIGHTS ARE CHANGED MALICIOUSLY!!!")
    return coef,intercept


def load_data(path,target,max_part):
    """
        path :  Path to the CSV 
        target: target feature 
        max_part: maximum number of partitions
        Assumption that it takes only cleaned data
        Loads the csv; Seperates the Y_varable;Partition the data; Return one Random Parition
    """

    data=pd.read_csv(path)
    data=data.sample(frac=1,random_state=rand(1,10000)) #Suffling
    y_var=data.pop(target)
    X,y= data.to_numpy(),y_var.to_numpy()
    print("**********************",len(X))
    
    return X,y
    


def set_initial_params(model,n_classes,n_features):

    model.classes_ = np.array([i for i in range(10)])
    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

def set_model_params(model, params):

    """Sets the parameters of a sklean LogisticRegression model."""

    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def get_model_parameters(model):
    global MALICIOUS
    if model.fit_intercept:
        params = [model.coef_,model.intercept_,]
        if(MALICIOUS):
            model.coef_,model.intercept_=mess_the_weights(model.coef_,model.intercept_)
            params = [model.coef_,model.intercept_,]
    else:
        params = [model.coef_,]
        if(MALICIOUS):
            model.coef_,model.intercept_=mess_the_weights(model.coef_,model.intercept_)
            params = [model.coef_]
    

    return params   

if __name__ == "__main__":
    global MALICIOUS,PROP
    MALICIOUS = False
    PROP = 0
    config = toml.load("../HiFlow3/config.toml")
    ID = int(sys.argv[1])
    if(config["EXPERIMENT"]["MALICIOUS"][ID-1] == 1):
        MALICIOUS = True
        PROP = config["EXPERIMENT"]["PROPORTION"] * 0.1
    X,y=load_data("../HiFlow3/node_data/"+sys.argv[1]+".csv","lable",2)#Here
    
    model = SGDClassifier(penalty="l2",max_iter=2,class_weight={0:6.464,1:0.542})
    set_initial_params(model,2,14)
    
    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X,y)
            return get_model_parameters(model), len(X), {}

        def evaluate(self, parameters, config):
            set_model_params(model, parameters)
            # loss = log_loss(Y_test, model.predict_proba(X_test))
            # accuracy = model.score(X_test, Y_test)
            return 0.61, len(X), {"accuracy": 55}

    # Start Flower client
    #fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())

    fl.client.start_numpy_client(server_address="0.0.0.0:29512", client=MnistClient())
