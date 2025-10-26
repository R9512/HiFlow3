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
        return False


def verifier(path,impli):
    """
        - Does actual verification.
    """
    dataset = path.split("/")[1]
    model_type = path.split("/")[-2]
    to_process = path+impli+"/EXP1_ZETA_STUFF/"
    #Flower also has multiple stuff
    gather_data = []
    for x in os.listdir(to_process):
        current = to_process+x+"/"
        zeta = int(x.split("=")[-1])
        a,deci=cross_checker(current,impli,model_type)
        if(impli == "HiFlow3" and deci):#All Good
            with open(current+"results.picke","rb") as f:
                data = p.load(f)
            final_data = {"DATASET":dataset,"MODEL_TYPE":model_type,"ZETA":zeta,"NUM_ZETA_MODELS":a,"IMPLIMENTATION":impli,"ACCURACY":data["ACCURACY"],
                          'VALIDATOR_GAS':data['VALIDATOR_GAS'],'COORDINATOR_GAS':data['COORDINATOR_GAS']}
            #file:///home/swami/Desktop/HiFlow3/HiFLOW3/DIABETICS/SGD/HiFlow3/EXP1_ZETA_STUFF/zeta=3/communication.pickle
            with open(current+"communication.pickle","rb") as f1:
                r = p.load(f1)
            network_data = pd.DataFrame(r)

            FINAL_NETWORK = {'IPFS': {'IN':0, 'OUT': 0},
                    'BLOCKCHAIN': {'IN': 0, 'OUT':0},
                    'COORDINATOR': {'IN': 0, 'OUT': 0},
                    'VALIDATORS': {'IN': 0, 'OUT': 0}}
            FINAL_ACCURACY = 0
            FINAL_COORDINATOR_GAS,FINAL_VALIDATOR_GAS =0,0

            for z in network_data:
                FINAL_NETWORK["IPFS"]["IN"]+= network_data[z]["IPFS"]["IN"]
                FINAL_NETWORK["IPFS"]["OUT"]+= network_data[z]["IPFS"]["OUT"]
                FINAL_NETWORK["BLOCKCHAIN"]["IN"]+= network_data[z]["BLOCKCHAIN"]["IN"]
                FINAL_NETWORK["BLOCKCHAIN"]["OUT"]+= network_data[z]["BLOCKCHAIN"]["OUT"]
                FINAL_NETWORK["COORDINATOR"]["IN"]+= network_data[z]["COORDINATOR"]["IN"]
                FINAL_NETWORK["COORDINATOR"]["OUT"]+= network_data[z]["COORDINATOR"]["OUT"]
                FINAL_NETWORK["VALIDATORS"]["IN"]+= network_data[z]["VALIDATORS"]["IN"]/5
                FINAL_NETWORK["VALIDATORS"]["OUT"]+= network_data[z]["VALIDATORS"]["OUT"]/5


            for z in FINAL_NETWORK:
                for x in FINAL_NETWORK[z]:
                    FINAL_NETWORK[z][x]/=MAX_ITER
            if(FINAL_NETWORK["VALIDATORS"]["OUT"]<0):
                print("IT is Less Tahn 0")

            final_data["C_IPFS"] = FINAL_NETWORK["IPFS"]["IN"] + FINAL_NETWORK["IPFS"]["OUT"]
            final_data["C_BLOCKCHAIN"] = FINAL_NETWORK["BLOCKCHAIN"]["IN"] + FINAL_NETWORK["BLOCKCHAIN"]["OUT"]
            final_data["C_SYSTEM"] = FINAL_NETWORK["COORDINATOR"]["IN"] + FINAL_NETWORK["COORDINATOR"]["OUT"]+ FINAL_NETWORK["VALIDATORS"]["IN"]
            gather_data.append(final_data.copy())

    return (pd.DataFrame(gather_data))

        #ACtual Process
def to_process(path):
    """
        - Does actual processing
        - Need to do check for LR and SGD, then count all are there
    """
    d = verifier(path+"SGD"+"/","HiFlow3")
    d1 = verifier(path+"SGD"+"/","Flower")
    d2 = verifier(path+"LR"+"/","HiFlow3")
    d3 = verifier(path+"LR"+"/","Flower")
    return(pd.concat([d,d1,d2,d3]))

print("adf11")
d = to_process("./DIABETICS/")
m = to_process("./MUSHROOM/")
t = to_process("./THYROID/")
FINAL_DATA = pd.concat([d,m,t])
FINAL_DATA.to_csv("./Visualization_Results/EXP1_RESULT.csv",index=False)
print("EXP1_Results.csv is generated in ./Visualization_Results/EXP1_RESULT.csv.. All the Best Visualization ")

