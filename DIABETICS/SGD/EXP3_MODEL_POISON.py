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
NUM_NODES = 15
#Need to change the TEST_PATH and GOURND TRUTH
TEST_PATH = "./HiFlow3/diabetics_data/test.csv"
TEST_DATA = pd.read_csv(TEST_PATH)
GROUND_TRUTH = TEST_DATA.pop("OUTCOME")

def guardrail():
    """
        - To Ensure the Python Script for Netwrok Monitoring is Measured
    """
    a = rand(999,9999)
    b = int(input(f"1.This deletes existing resutls \n 2.Need to change the TEST_PATH and GOURND TRUTH\n If you are sure ,Enter {a}:"))
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
    check = "HiFloW3"
    if(type == "Flower"):
        check = ".pickle"
        #process the flower things
    ACC = []
    FINAL_ACCURACY = 0
    print("reciebved path",path)
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

        
            


def mopper(flip,prop):
    """ 
        MOVES AROUND THE STUFF
        - Flower:
    """
    base = "./Flower/"
    source = base+"EXP3_MODEL_POISON/"+str(flip)+","+str(prop)
    os.mkdir(source)
    #HiFlow3 movel all .pickle to it
    for x in os.listdir(base):

        if(".pickle" in x):
            print(x)
            move(base+x,source)


    base = "./HiFlow3/"
    source = base+"EXP3_MODEL_POISON/"+str(flip)+","+str(prop)
    os.mkdir(source)
    #HiFlow3 movel all .pickle to it
    for x in os.listdir(base):
        if(".pickle" in x):
            move(base+x,source)
    print(source)
    tester("HiFlow3","HiFlow3/EXP3_MODEL_POISON/"+str(flip)+","+str(prop))
    tester("Flower","Flower/EXP3_MODEL_POISON/"+str(flip)+","+str(prop))
    #Remove TXTs from here


def auto_model_poisoning():
    """
        - Generic Functions that make things work

    """
    global TEST_DATA,GROUND_TRUTH
    os.mkdir("./HiFlow3/EXP3_MODEL_POISON")
    os.mkdir("./Flower/EXP3_MODEL_POISON")

    #just setting situations for Exp2
    for flip in range(0,4):
        for prop in range(2,7,2):#20,
            config = toml.load("./HiFlow3/config.toml")
            config["EXPERIMENT"]["FLIP_NUM"] = 0#Dont Worry It should be zero; Or partitioner will go crazy
            config["EXPERIMENT"]["PROPORTION"] = prop
            t_flip = flip*3#num nodes to be spoiled
            malicious = [0 for x in range(NUM_NODES)]
            count = 0
            while count < t_flip:
                t = rand(0,NUM_NODES-1)
                if(malicious[t] == 0):
                    malicious[t] = 1
                    count+=1
            config["EXPERIMENT"]["MALICIOUS"] = malicious.copy()
            with open("./HiFlow3/config.toml", 'w') as f:
                toml.dump(config, f)
            executor()
            mopper(flip,prop)

#CAREFULL
guardrail()
config = toml.load("./HiFlow3/config.toml")
config["EXPERIMENT"]["A_TYPE"] = "MODEL"
config["EXPERIMENT"]["FLIP_NUM"] = 0
config["EXPERIMENT"]["PROPORTION"] = 0
config["EXPERIMENT"]["MALICIOUS"] = [ 0,0,0, 0, 0,]
config["COORDINATOR"]["ZETA"] = 2
with open("./HiFlow3/config.toml", 'w') as f:
    toml.dump(config, f)

rmtree("./HiFlow3/EXP3_MODEL_POISON",ignore_errors=True)
rmtree("./Flower/EXP3_MODEL_POISON",ignore_errors=True)
#Detele All .txt or .pickle files
for x in os.listdir("./HiFlow3/"):
    if(".txt" in x or ".pickle" in x):
        os.remove("./HiFlow3/"+x)
for x in os.listdir("./Flower/"):
    if(".txt" in x or ".pickle" in x):
        os.remove("./Flower/"+x)
auto_model_poisoning()
