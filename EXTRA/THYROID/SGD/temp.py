import os
import pickle as p
import pandas as pd
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
for x in range(0,5):
    base_path = "./HiFlow3/EXP2_DATA_POISON/"
    for x in os.listdir(base_path):
        print(x)
        temp = base_path+x
        with open(temp+"/results.picke","rb") as f:
            data = p.load(f)
        print("\tFLow3:",data["ACCURACY"])
        temp = base_path+x
        with open(temp+"/results.picke","rb") as f:
            data = p.load(f)
        print("\tFlower:",data["ACCURACY"])
    print("================")
