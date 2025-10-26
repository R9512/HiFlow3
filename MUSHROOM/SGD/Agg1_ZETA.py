"""
    - Just Aggregate Stuff 
"""
import os
import pickle as p
import pandas as pd
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
#for x in range(0,5):
    #base_path = "./HiFlow3/EXP1_ZETA/zeta="+str(x)
    #with open(base_path+"/results.picke","rb") as f:
        #data = p.load(f)
    #data["ZETA"] = x
    #print("FLow3:",data["ACCURACY"],data["A_LIST"])
    #base_path = "./Flower/EXP1_ZETA/zeta="+str(x)
    #with open(base_path+"/results.picke","rb") as f:
        #data = p.load(f)
    #data["ZETA"] = x
    #print("Flower:",data["ACCURACY"])
TEST_PATH = "./HiFlow3/mush/test.csv"
TEST_DATA = pd.read_csv(TEST_PATH)
GROUND_TRUTH = TEST_DATA.pop("class")

for x in range(0,5):
    base_path = "./HiFlow3/EXP1_ZETA/zeta="+str(x)
    count = 0
    z_acc = 0
    n_acc = 0
    print("Zeta:",x)
    for z in os.listdir(base_path):
        if("HiFloW3" in z):
            with open(base_path+"/"+str(z),"rb") as f:
                    model = p.load(f)
            prediction = model.predict(TEST_DATA)
            acc = accuracy_score(GROUND_TRUTH,prediction)
            if(z[0]=="Z"):
                count+=1
                z_acc += acc
            else:
                n_acc+=acc

        if(count == 0):
            count = 1
    print("Normal:",n_acc/(10-count),"Zeta:",z_acc/count )
