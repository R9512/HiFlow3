from os import listdir 
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import LogisticRegression
import pickle,sys
#1 is logisticregression;2 is LogisticRegression

NUM_MODELS = 10#How many should be there
BASE_PATH = "./FINAL_MODELS"
BASE_MODEL = LogisticRegression() if sys.argv[1] == '1' else  LogisticRegression()
print("PLEASE NOTE THAT IF NO OUTPUT THEN ALL IS GOOD \n 1: For SGD 0: For LogisticRegression")
for x in listdir(BASE_PATH):
    zeta_path = BASE_PATH + "/"+x+"/"
    for x1 in listdir(zeta_path):
        processing = zeta_path+x1+"/"
        pickle_counter = 0
        not_same_asked=[]
        for x2 in listdir(processing):
            if ".pickle" in x2:
                pickle_counter+=1
                checking = processing+x2
                with open(checking,"rb") as f:
                    model = pickle.load(f)
                    if type(model) != type(BASE_MODEL):
                        not_same_asked.append(checking)
            

        if pickle_counter != NUM_MODELS:
            print("[CRITICAL] PATH:",processing,":: To Repeat:",NUM_MODELS - pickle_counter)
        if len(not_same_asked) !=0:
            for x3 in not_same_asked:
                print("\t[CRITICAL] Model Mismatch:",x3)
        
