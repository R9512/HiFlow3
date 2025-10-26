import os
import pickle as p
import pandas as pd
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
for x in range(0,5):
    base_path = "./HiFlow3/EXP1_ZETA/zeta="+str(x)
    with open(base_path+"/results.picke","rb") as f:
        data = p.load(f)
    data["ZETA"] = x
    print("\tFLow3:",data["ACCURACY"])
    base_path = "./Flower/EXP1_ZETA/zeta="+str(x)
    with open(base_path+"/results.picke","rb") as f:
        data = p.load(f)
    data["ZETA"] = x
    print("\tFlower:",data["ACCURACY"])
    print("================")
