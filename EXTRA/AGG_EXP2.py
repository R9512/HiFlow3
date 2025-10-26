import pickle as p
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
import os
import pandas as pd

"""
    - Looking for aggregating all the Experiment resutls
    - Goes through each thing and prints stuff
RECORD:
DATASET
MODEL_TYPE
ZETA
NUM_ZETA
C_IPFS
C_BLOCKCHAIN
C_DA
C_VALIDATOR
G_DA
G_VALIDATOR
NUM_OC_GEN_MODELS #SC Generated Model
HiFlow3_ACC
FLower_ACC
"""
MAX_ITER = 10
def cross_checker(path,impli,model_type):
    """
        - DOes all the checking
        - Ensure there are 10 entries of same type of the type
    """
    count = 0
    check_pharse = "HiFloW3"
    m_type = SGDClassifier
    zeta_count = 0
    if(impli == "Flower"):
        check_pharse = "FL"
    if(model_type == "LR"):
        m_type=LogisticRegression

    for x in os.listdir(path):

        current = path+x
        if(check_pharse in x):
            #Need to check the implmentation
            with open(current,"rb") as f:
                model = p.load(f)

            if(isinstance(model,m_type)):
                count+=1
                if("Z" in x):
                    zeta_count+=1
    if(count == MAX_ITER):
        return zeta_count,True
    else:
        print("Some Issue with:",path,impli,model_type,"Count:",count)
        return  zeta_count,False


def verifier(path,impli):
    """
        - Does actual verification.
    """
    dataset = path.split("/")[1]
    model_type = path.split("/")[-2]
    to_process = path+impli+"/EXP2_DATA_POISON_STUFF/"
    #Flower also has multiple stuff
    gather_data = []
    stuff = []
    for x in os.listdir(to_process):
        stuff.append(x)
        current = to_process+x+"/"
         
        a,deci=cross_checker(current,impli,model_type)
        if(deci):#All Good
            with open(current+"results.picke","rb") as f:
                data = p.load(f)
            final_data = {"DATASET":dataset,"MODEL_TYPE":model_type,"ZETA":2,"NUM_ZETA_MODELS":a,
                            "IMPLIMENTATION":impli,"FLIP,PROP":x,"ACCURACY":data["ACCURACY"]}
            #file:///home/swami/Desktop/HiFlow3/HiFLOW3/DIABETICS/SGD/HiFlow3/EXP1_ZETA_STUFF/zeta=3/communication.pickle
            
        gather_data.append(final_data.copy())
    if(len(stuff) != 12):
        print("SOMTHING IS WRONG WITH",path,impli,"Couldnt Find All 12")
        print(stuff)

    return (pd.DataFrame(gather_data))

        #ACtual Process
def to_process(path):
    """
        - Does actual processing
        - Need to do check for LR and SGD, then count all are there
    """
    d = verifier(path+"SGD"+"/","HiFlow3")
    d1 = verifier(path+"SGD"+"/","Flower")

    return(pd.concat([d,d1]))

print("adf11")

t = to_process("./THYROID/")
FINAL_DATA = pd.concat([t])
FINAL_DATA.to_csv("./Visualization_Results/EXP2_RESULT.csv",index=False)
print("EXP2_Results.csv is generated in ./Visualization_Results/EXP2_RESULT.csv.. All the Best Visualization ")

