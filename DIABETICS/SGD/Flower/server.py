import flwr as fl
from sklearn.metrics import log_loss
from sklearn.linear_model import SGDClassifier
from typing import Dict
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as split
from random import randint as rand
import toml
from shutil import rmtree
from os import mkdir
from datetime import datetime as dt
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
    X_train,X_test,Y_train,Y_test=split(data,y_var,test_size=0.3)
    X,y= X_train.to_numpy(),Y_train.to_numpy()
    partitions = list(zip(np.array_split(X, max_part), np.array_split(y, max_part)))
    single_partition = partitions[np.random.choice(max_part)]
    part_x,part_y=single_partition
    return part_x,X_test,part_y,Y_test

def initialize_parameter():
    global C_PRI,C_PUB,CON_ADD,NUM_VALIDATORS,NUM_PARITIONS,PATH,TARGET,UNIFORM,VARIENCE,ZETA
    global NAME   
    config = toml.load("config.toml")
    C_PRI,C_PUB,CON_ADD,NUM_VALIDATORS,NUM_PARITIONS,PATH,TARGET,UNIFORM,VARIENCE,ZETA,_ = config["COORDINATOR"].values()
    
    return PATH
    


def data_prep():
    global NUM_VALIDATORS,NUM_PARITIONS,PATH,TARGET,UNIFORM,VARIENCE

    data = pd.read_csv(PATH)
    data=data.drop_duplicates(keep=False)
    data=data.sample(frac=1)
    data_len = data.shape[0]
    mini = data_len//(NUM_PARITIONS)
    maxi = mini+VARIENCE
    PARTITIONS=[]
    while (data.shape[0]>maxi) :
        
        if not UNIFORM and data.shape[0]>maxi:
            maxi=mini+rand(10,VARIENCE)
            part = data.sample(n=rand(mini-VARIENCE,maxi))

        elif(data_len//(NUM_PARITIONS) < data.shape[0]):
            maxi=data_len//(NUM_PARITIONS)
            part = data.sample(n=data_len//(NUM_PARITIONS))
        temp = pd.concat([part,data])
        data=temp.drop_duplicates(keep=False)
        PARTITIONS.append(part)
    
    if not UNIFORM:

        if(data.shape[0]>mini):
            PARTITIONS.append(data[:mini])
        while len(PARTITIONS)>NUM_PARITIONS:
            PARTITIONS.pop()
    else:
        while len(PARTITIONS) != NUM_PARITIONS:
            PARTITIONS.append(data[:mini])
            data=data[:mini]
    rmtree("data",ignore_errors=True)
    mkdir("data")
    config = toml.load("config.toml")
    FLIP,PROPORTION = config["EXPERIMENT"]["FLIP_NUM"],config["EXPERIMENT"]["PROPORTION"]
    for x in range(0,FLIP):
        col = config["COORDINATOR"]["TARGET"]
        data=PARTITIONS[x]
        print("Before",data[col].value_counts())
        df_1 = data[data[col]==1].reset_index()
        df_0 = data[data[col]==0].reset_index()
        fraction_0,fraction_1 =int(df_0.shape[0]*PROPORTION),int(df_1.shape[0]*PROPORTION)
        df_1.loc[1:fraction_1,col] = 0
        df_0.loc[1:fraction_0,col] = 1
        data_frame = pd.concat([df_1,df_0])
        data_frame.pop("index")
        print("\nAfter",data_frame[col].value_counts())
        PARTITIONS[x] = data_frame
    for y,x in enumerate(PARTITIONS):
        x.to_csv("data/"+str(y)+".csv",index=False)
    with open("done","w") as f:
        f.write("CREATED PARTITIONS")
    print("Generated Partitions")

#==============================

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


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, X_test,_, y_test = load_data("./diabetics_data/test.csv","OUTCOME",2)


    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        global NAME
        set_model_params(model, parameters)
        #loss = log_loss(y_test, model.predict_proba(X_test.values))
        loss = 0
        accuracy = model.score(X_test.values, y_test)

       
        if(server_round == 5):
            print("############Saving MOdel") 
            n = "classic_FL_model"+str(dt.now().strftime("%Y_%m_%d_%H_%M_%S"))+".pickle"
            pickle.dump(model, open(n, 'wb'))#change name here put time bro 
           
            with open("done_part.flag","w") as f:
                f.write("")
        
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    #Data prep goes here
    initialize_parameter()
    model = SGDClassifier(penalty="l2",max_iter=2,class_weight={1:1.433,0:0.768})
    set_initial_params(model,2,21)
    strategy = fl.server.strategy.FedAvg(min_available_clients=15,evaluate_fn=get_evaluate_fn(model),on_fit_config_fn=fit_round,)
    hist =fl.server.start_server(server_address="0.0.0.0:29512",strategy=strategy,config=fl.server.ServerConfig(num_rounds=5,))
    
