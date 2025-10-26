from os import listdir 
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import LogisticRegression
from tomli import load
from sklearn.metrics import accuracy_score
import pickle,sys
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
#1 is logisticregression;2 is LogisticRegression
#Finding the test path
DATASET = []
MODEL_TYPE = []
FLIP = []
PROP = []
ZETA = []
ACCURACY = []
TYPE = []#Will always be Flow3

def setup_test():
    """
        setup the test dataset
    """
    with open("config.toml","rb") as f:
        data = load(f)
    data = data["COORDINATOR"]
    TARGET = data["TARGET"]
    dataset = {"lable":"THYROID","class":"MUSHROOM","OUTCOME":"DIABETICS"}
    train_path = data["PATH"]
    test_path = train_path.split("train")[0]+"test.csv"
    test_data = pd.read_csv(test_path)
    ground_truth = test_data.pop(data["TARGET"])
    return (test_data,ground_truth,dataset[TARGET])

def appender(dataset, path,acc,model_type):
    """
        appends the list and its values
    """
    
    base = path.split("=")[1]
    zeta = int (base[0])
    flip = int (base[2])
    prop = int (base[4])
   
    ZETA.append(zeta)
    FLIP.append(flip)
    PROP.append(prop)
    DATASET.append(dataset)
    TYPE.append("FLOW3")
    ACCURACY.append(acc)
    MODEL_TYPE.append(model_type)
    


TEST_DATA, GROUND_TRUTH,CURRENT_DATASET = setup_test()
NUM_MODELS = 10#How many should be there
BASE_PATH = "./FINAL_MODELS"
BASE_MODEL = LogisticRegression()
base_model_type = ""
for x in listdir(BASE_PATH):
    zeta_path = BASE_PATH + "/"+x+"/"
    for x1 in listdir(zeta_path):
        processing = zeta_path+x1+"/"
        pickle_counter = 0
        not_same_asked=[]
        for x2 in listdir(processing):
            if ".pickle" in x2:
                #print("FOUDNNNND")
                pickle_counter+=1
                checking = processing+x2
                with open(checking,"rb") as f:
                    model = pickle.load(f)
                    if type(model) == type(BASE_MODEL):
                        model_type = "LogisticRegression"
                    else:
                        model_type = "LogisticRegression"
                    pred = model.predict(TEST_DATA)
                    acc = accuracy_score(GROUND_TRUTH,pred)
                    appender(CURRENT_DATASET,processing, acc,model_type)
                   
actual_data = {"DATASET":DATASET,"TYPE":TYPE,"MODEL_TYPE":MODEL_TYPE,
                "ZETA":ZETA,"FLIP":FLIP,"PROP":PROP,"ACCURACY":ACCURACY}
ACCURACY_DATA = pd.DataFrame.from_dict(actual_data)   
FINAL_DATA = []
for zeta in range(0,1):
    for flip in range(0,4):
        for prop in range(0,10,2):
            d = ACCURACY_DATA.query("ZETA == "+str(zeta)+" & FLIP == "+str(flip)+" & PROP == "+str(prop))
            #print(d)
            #print(d.shape)
            print("*/*/*/*/*/"*5)
            if d.shape[0] == 10:
                base = d.iloc[1].copy()
                base.ACCURACY = d.ACCURACY.mean()      
                base["STD"] = d.ACCURACY.std()
                base = base.to_dict()
                FINAL_DATA.append(base)
            else:
                print(d.shape)
                print("Threshold of Models Not Found: Zeta, FLIP,Prop",zeta, flip, prop)
                print("\t Remaining:",10-d.shape[0])

AGG_DATA = pd.DataFrame(FINAL_DATA)
AGG_DATA.to_csv("final_info.csv",index=False)
print(AGG_DATA)

        
    #print(processing)
        
