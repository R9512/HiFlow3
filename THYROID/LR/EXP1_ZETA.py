"""
Experiment 1: Count the netwrok stats,GAS,  for Flower and HiFLow3 for various values of Zeta
    Add an akser to enusre the netwrok count is ready
    Count: 
        - HiFlow3 in pickle then it reached all the things
    - This scipt is used to automate the experiment
    After reaching the maximum number of repetations, it should:
        1. in Case of HiFlower:
            a. Organize the files based on the Zeta Value
            b. Find the average of accuracy of the models. and Average them
            c. Average Gas Usage for validator and coordinator 
            d. Average of the Network usage
"""
from random import randint as rand
import os
import toml
from shutil import rmtree
from shutil import move
import pickle as p
import pandas as pd
from sklearn.metrics import accuracy_score
MAX_ITER = 10
#Need to change the TEST_PATH and GOURND TRUTH
TEST_PATH = "./HiFlow3/thyroid_data/test.csv"
TEST_DATA = pd.read_csv(TEST_PATH)
GROUND_TRUTH = TEST_DATA.pop("lable")

def guardrail():
    """
        - To Ensure the Python Script for Netwrok Monitoring is Measured
    """
    a = rand(999,9999)
    b = int(input(f"1.This deletes existing resutls \n 2.Need to change the TEST_PATH and GOURND TRUTH\n 3.If you are sure, SCAPY based Network Monitoring is up(Ensure it is reset) ,Enter {a}:"))
    if(a!=b):
        exit(0)


def counter(path):
    """
        - Counts the pickles generated
    """
    counter = 0
    if(path == "HiFlow3"):
        for x in os.listdir("./"+path+"/"):
            if("HiFloW3.pickle" in x):
                counter+=1
    if(path == "Flower"):
        for x in os.listdir("./"+path+"/"):
            if(".pickle" in x):
                counter+=1
    return counter

def executor():
    """
        - Acutal Executor 
        - Rule: HiFlow3 followed by Flower
        - Ensure the Counters are done
    """
    hiflow_count,flower_count = 0,0
    prev_h,prev_f = 0,0#initializing the previous counter
    while hiflow_count!=MAX_ITER and flower_count!=MAX_ITER:
        prev_h,prev_f = counter("HiFlow3"),counter("Flower")#initnal count
        os.system("cd ./HiFlow3 && ./try.sh")
        a = counter("HiFlow3")
        if(a == prev_h):
            continue
        else:
            hiflow_count+=1
        os.system("cd Flower && ./run.sh")
        b = counter("Flower")
        if(b == prev_f):
            continue
        else:
            flower_count+=1

    print("Implemented Max Iter")

def tester(type,path):
    """
        - Flower:
            - Pickle model run it through Test path
    """
    global TEST_DATA
    if(type == "Flower"):
        pass
        #process the flower things
    if(type == "HiFlow3"):
        print("To Process:",path)
        print("Here is the stuff, process the netwrok")
        with open(path+"/communication.pickle","rb") as f:
                    network_data = p.load(f)
        print(network_data)
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
        print(path)
        c_gas = 0
        v_gas = 0
        ACC = []
        for x in os.listdir(path):
            if("HiFloW3.pickle" in x):#it is a model
                print(x)
                with open(path+"/"+x,"rb") as f:
                    model = p.load(f)
                print(model)
                prediction = model.predict(TEST_DATA)
                acc = accuracy_score(GROUND_TRUTH,prediction)
                ACC.append(acc)
                FINAL_ACCURACY+=acc
            if(".txt" in x):#It is the gas thing; Need to take average of it
                with open(path+"/"+x, 'r') as file:
                        content = file.read()
                
                if("C" in x):#It is coordinator
                    gas = content.split(",")
                    gas = [int(x) for x in gas if len(x) != 0]
                    FINAL_COORDINATOR_GAS=sum(gas)

                else:
                    gas = content.split(";")
                    gas = [int(x) for x in gas if len(x) != 0]
                    FINAL_VALIDATOR_GAS+=sum(gas)
        FINAL_COORDINATOR_GAS/=MAX_ITER
        FINAL_VALIDATOR_GAS/=5*MAX_ITER
        FINAL_ACCURACY = FINAL_ACCURACY/MAX_ITER
        FINAL_NETWORK["COORDINATOR_GAS"] = FINAL_COORDINATOR_GAS
        FINAL_NETWORK["VALIDATOR_GAS"] = FINAL_VALIDATOR_GAS
        FINAL_NETWORK["ACCURACY"] = FINAL_ACCURACY
        FINAL_NETWORK["A_LIST"] = ACC.copy()
        with open(path+"/results.picke","wb") as f:
            p.dump(FINAL_NETWORK,f)
    if(type == "Flower"):
        ACC = []
        FINAL_ACCURACY = 0
        for x in os.listdir(path):    
            if(".pickle" in x):#it is a model
                print(x)
                with open(path+"/"+x,"rb") as f:
                    model = p.load(f)
                print(model)
                prediction = model.predict(TEST_DATA)
                acc = accuracy_score(GROUND_TRUTH,prediction)
                ACC.append(acc)
                FINAL_ACCURACY+=acc
        data = {"ACCURACY":FINAL_ACCURACY/MAX_ITER,"A_LIST":ACC.copy()}
        with open(path+"/results.picke","wb") as f:
            p.dump(data,f)

        
            


def mopper(zeta):
    """ 
        - Flower:
    """
    print("SAIRM")
    base = "./HiFlow3/"
    source = base+"EXP1_ZETA/zeta="+str(zeta)
    os.mkdir(source)
    #HiFlow3 movel all .pickle to it
    for x in os.listdir(base):

        if(".pickle" in x or ".txt" in x):
            print(x)
            move(base+x,source)

    base = "./Flower/"
    source = base+"EXP1_ZETA/zeta="+str(zeta)
    os.mkdir(source)
    #HiFlow3 movel all .pickle to it
    for x in os.listdir(base):
        if(".pickle" in x or ".txt" in x):
            move(base+x,source)
    tester("HiFlow3","HiFlow3/EXP1_ZETA/zeta="+str(zeta))
    tester("Flower","Flower/EXP1_ZETA/zeta="+str(zeta))


def auto_zeta():
    """
        - Generic Functions that make things work

    """
    global TEST_DATA,GROUND_TRUTH
    #just setting situations for Exp1
    config = toml.load("./HiFlow3/config.toml")
    
    config["EXPERIMENT"]["A_TYPE"] = "NONE"
    config["EXPERIMENT"]["FLIP_NUM"] = 0
    config["EXPERIMENT"]["PROPORTION"] = 0
    config["EXPERIMENT"]["MALICIOUS"] = [ 0 for x in range(0,15)]
    os.mkdir("./HiFlow3/EXP1_ZETA")
    os.mkdir("./Flower/EXP1_ZETA")
    with open("./HiFlow3/config.toml", 'w') as f:
        toml.dump(config, f)
    for x in range(0,6):
        config = toml.load("./HiFlow3/config.toml")
        config["COORDINATOR"]["ZETA"] = x
        with open("./HiFlow3/config.toml", 'w') as f:
            toml.dump(config, f)
        executor()
        mopper(x)

#CAREFULL
guardrail()
# a = counter("HiFlow3")
# b = counter("Flower")
# print(a,b)
#executor()
rmtree("./HiFlow3/EXP1_ZETA",ignore_errors=True)
rmtree("./Flower/EXP1_ZETA",ignore_errors=True)
auto_zeta()
