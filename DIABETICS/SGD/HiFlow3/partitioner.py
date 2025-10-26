"""
    - Use this code for partitioning the data.
    - Relies on flwr datasets
    - Just need to create the list with variences and uniform options.
    - LETS DO EVEYRTHING MANUALLY.....
"""
from os import mkdir
from shutil import rmtree
import pandas as pd
import numpy as np
import tomli, tomli_w

import random
import warnings
warnings.filterwarnings("ignore")

ROLE = "AGGREGATOR"
def preprocessor(config):
    """
        - Takes a config dictionary and returns a list of categorical values
        - If anything tha has less than 5 distinct values is considered as categorical variable.
            #Ensure this fits with the data whatever you are woking on it.
        - Bascially determines which is categorical variable and which is not.
    """
    
    NUM_PARITIONS = config["COORDINATOR"]["NUM_VALIDATORS"]
    UNIFORM = config["COORDINATOR"]["UNIFORM"]
    TARGET = config["COORDINATOR"]["TARGET"]
    VARIENCE = config["COORDINATOR"]["VARIENCE"]
    A_TYPE,NUM_MAL,MAL_PROP,_ = config["EXPERIMENT"].values()
    MAL_PROP*=0.1#Carefull
    print(NUM_MAL,"FROM PARTITION")
    
    data = pd.read_csv(config["COORDINATOR"]["PATH"])
    data=data.drop_duplicates(keep=False)
    data=data.sample(frac=1)
    data_len = data.shape[0]
    GROUND_TRUTH = data[TARGET].unique().tolist()
    mini = data_len//(NUM_PARITIONS)
    maxi = mini+VARIENCE
    cat_varaible = {}
    PARTITIONS = []
    for x in data.columns:
        if(data[x].value_counts().shape[0] < 5):
            cat_varaible[x] = data[x].unique().tolist()
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
    # temp = "Value",A_TYPE
    # logging(temp,ROLE)
    if(A_TYPE == "DATA" or A_TYPE == "BOTH"):
        if(NUM_MAL > NUM_PARITIONS):
            raise Exception("[NO GOOD DATA IS FOUND]Num Malicious is larger than that of Num Partitions")
           
        for x1 in range(0,NUM_MAL):
            data=PARTITIONS[x1].copy()

            before = data[TARGET].value_counts()
            lables = data.pop(TARGET).to_list()
            TO_CHANGE = int(len(lables) * (MAL_PROP))
            for x in range(0,TO_CHANGE):
                r = random.choice(GROUND_TRUTH)
                while r == lables[x]:
                    r = random.choice(GROUND_TRUTH)
                lables[x] = r #Actual chaning
            #Doing the feature flipping
            mask = np.random.choice([-1,0,1],size = (TO_CHANGE,data.shape[1]),p=[MAL_PROP/2,1-MAL_PROP,MAL_PROP/2])
            mask_df = pd.DataFrame(mask, index=data.index[:TO_CHANGE], columns=data.columns)
            data[mask_df == 1] = 255#Creating white plots
            data[mask_df == -1] = 0 #Creating black plots
            data[TARGET] = lables#Adding the 
            after = data[TARGET].value_counts()
            after = pd.merge(after,before,on=TARGET,how="inner")
            after = after.reset_index()
            after["diff"] = after["count_x"] - after["count_y"]
            print("Changes Made: For \n"+str(after["diff"]))
            print("Total Num of Records"+str(after["count_x"].sum())+";"+str(after["count_y"].sum()))
            data = data.sample(frac=1)#Shouyld come last
            PARTITIONS[x1] = data
            print("=*="*30)
  
        
    rmtree("data",ignore_errors=True)
    mkdir("data")
    for x in range(0,NUM_MAL):
        PARTITIONS[x].to_csv("./data/"+str(x)+"_m.csv",index=False)
    for x in range(NUM_MAL,len(PARTITIONS)):
        PARTITIONS[x].to_csv("./data/"+str(x)+".csv",index=False)
    return config


#MAIN FUNCTION
with open("config.toml","rb") as f:
    CONFIG = tomli.load(f)
print("Starting the Partitioner")
CONFIG = preprocessor(CONFIG)


print("Partitioning Done",ROLE)
