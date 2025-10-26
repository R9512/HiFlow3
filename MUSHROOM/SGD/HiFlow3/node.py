import socketio
import warnings,sys
import pandas as pd
import json
import gnupg
import toml
import ipfs_api as ips
from time import sleep
from sys import exit as exit
from sklearn.model_selection import train_test_split as split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss as loss
from sklearn.metrics import accuracy_score as acc
from datetime import datetime as dt
from random import randint as rand
from os import remove
import numpy as np
sio = socketio.Client()
GX1,GY1=0,0#Data Holder
N_CLASSES,N_FEATURES=0,0
NUM_DATA_RECORDS = 0
MODEL=0
GPG_OBJ = gnupg.GPG(gnupghome='/home/swami/.gnupg/')
GPG_OBJ.encoding = 'utf-8'

#===================================
def mess_the_weights(coef,intercept):
    """
        - Changes the weights and intecept
        - Follows hybrid appraoch, like adding noise and change flipping
        - First adding the noise then flipping the changes
    """
    global PROP

    #Adding Noise
    noise_coef = np.random.normal(loc=0, scale=PROP, size=coef.shape)
    coef+=noise_coef
    noise_intercept = np.random.normal(loc=0, scale=PROP, size=intercept.shape)
    intercept+=noise_intercept
    #Signing Flipping
    n = coef.size
    k = int(PROP*n)
    coef_mask = np.random.choice(n,size=k,replace=False)
    coef_mask = np.unravel_index(coef_mask,coef.shape)
    coef[coef_mask]*=-1
    n = intercept.size
    k = int(PROP*n)
    #print("BEFORE:",intercept)
    intercept_mask = np.random.choice(n,size=k,replace=False)
    intercept_mask = np.unravel_index(intercept_mask,intercept.shape)
    intercept[intercept_mask]*=-1
    # print("AFTER:",intercept)
    # print("="*6)
    print("MODEL WEIGHTS ARE CHANGED MALICIOUSLY!!!")
    return coef,intercept


def payload_parser(payload):
    """
        - Parses the pay load
        - Split the Data
    """
    global ID
    payload=json.loads(payload)
    data = json.loads(payload["data"])
    data =pd.DataFrame(data)
    data.to_csv("./node_data/"+str(ID)+".csv",index=False)#Saving the model
    target = data.pop(payload["target"])
    payload.pop("data")
    return (data,target,payload)

def payload_sender(names,values,eventname):
    """
        - Just send the payload
    """
    global GPG_OBJ, VALIDATOR
    if(len(names)!=len(values)):
        raise TypeError("Lengths of the names and values are not same")
    payload = {}
    for i,x in enumerate(names):
        payload[x]=values[i]
    payload_enc = json.dumps(payload)
    encrypted_data = GPG_OBJ.encrypt(payload_enc,VALIDATOR)
    name = str(rand(0,5000))+"_"+str(dt.now())+".txt"
    name = "./data/"+name
    with open(name,"w") as file:
        file.write(encrypted_data.data.decode())
    cid = ips.publish(name)
    #print("DATA IS SENT TO THE IPFS. Just Delete the file",cid)
    remove(name)

    #Add IPFS_HERE
    #sio.emit(eventname,payload)
    sio.emit(eventname,cid)



def get_model_parameters(new_model,old_model):
    """
        x= 23.5; After traing;x1=56.4;update =

        * NEW MODEL - OLD MODEL

        - Returns the update of the model
        - Expects two dicts, each dicts has the weights and bias of a model
        - 1 st argument should be new one
        - 2 nd is old one


    #Fix the values here, like the float - int is coming here..
    """
    #params = [model.coef_.tolist(),model.intercept_.tolist(),]
    updates = {"weights":[],"bias":[]}
    for x in range(0,len(old_model["weights"][0])):
        updates["weights"].append(new_model["weights"][0][x]-old_model["weights"][0][x])

    updates["bias"] = [new_model["bias"][0]-old_model["bias"][0]]
    #print("These are the Updaes",updates)
    return updates


def set_model_parameters(data):
    """
        - This is only for the Logistic Regression
        - The built platform is NOT model agnostic
    """
    global MODEL

    weights,bias = data["weights"],data["bias"]
    MODEL.coef_ = np.array([weights])
    if MODEL.fit_intercept:
        MODEL.intercept_ = np.array([bias])


def prepare_initial_model(weights,bias):
    """
    Preapares the Inintial Model only
        - Currenlty Starting with all 0s
    """
    #print("In tghe weights",weights,bias)
    model = SGDClassifier(penalty="l2",max_iter=2,warm_start=True)
    model.coef_ = np.array(weights)

    if model.fit_intercept:
        model.intercept_ = np.array(bias)
    return model



def model_fitter():
    global MODEL,GX1,GY1

    #print("\t\tInside the model_fitter")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        MODEL.fit(GX1,GY1)
        #rint("Not Here is the problem")
    if(MALICIOUS):
        MODEL.coef_,MODEL.intercept_ = mess_the_weights(MODEL.coef_,MODEL.intercept_)




def controller(data):
    """
    Controls the Actions HopeFully
    """
    global GX1,GY1,NUM_DATA_RECORDS,MODEL
    GX1,GY1,payload = payload_parser(data)
    N_CLASSES,N_FEATURES=payload["n_classes"],payload["n_features"]
    NUM_DATA_RECORDS=payload["num_data_records"]
    #
    MODEL = prepare_initial_model(payload["genesis_model"]["weights"],payload["genesis_model"]["bias"])

    model_fitter()
    #Call The Function Which Will Compute the Update
    payload_sender(["round_num","weights","num_data_records"],
                    [payload["round_num"],get_model_parameters({"weights":MODEL.coef_.tolist(),"bias":MODEL.intercept_.tolist()},payload["genesis_model"]),payload["num_data_records"]],
                    "update_round")






#======================
@sio.event
def recieve_data(data):
    controller(data)

@sio.on('my message')
def on_message(data):
    print('I received a message!')

@sio.event
def connect():
    print("[NODE]Connecting To Validator")

    print("[NODE]Connected Just Now. Emiting the ready signal")
    sio.emit("ready")



@sio.event
def fl_round(data):

    global GX1,GY1,MODEL

    set_model_parameters(data)
    OLD_MODEL_DATA ={"weights":[data["weights"]],"bias":[data["bias"]]}
    #print("THE DATA:::",OLD_MODEL_DATA)
    model_fitter()

    payload_sender(["round_num","weights","num_data_records"],
                        [data["round_num"],get_model_parameters({"weights":MODEL.coef_.tolist(),"bias":MODEL.intercept_.tolist()},OLD_MODEL_DATA),NUM_DATA_RECORDS],
                        "update_round")
    #print("Round No,Loss,Accuracy=>",data["round_num"],loss,accuracy)



@sio.event
def on_disconnect(data):
    print("[NODE]Disconnect Message From Server:",data)
    sio.disconnect()
    sio.shutdown()

global VALIDATOR, ID, MALICIOUS
config = toml.load("config.toml")
print(config["VALIDATOR"]["MIN_CLIENTS"])
portnum,id=sys.argv[1],int(sys.argv[2])
MAN_ID = int(portnum) - 44600
MALICIOUS = False
PROP = 0
ID = id
VALIDATOR = "VALIDATOR_"+str(int(portnum) - 44600)+"@HiFlow3.com"
if((config["EXPERIMENT"]["A_TYPE"] == "MODEL" or config["EXPERIMENT"]["A_TYPE"] == "MODEL" == "BOTH")):
    if(config["EXPERIMENT"]["MALICIOUS"][ID-1] == 1):#It means the manager is malicious
        MALICIOUS = True
        PROP = config["EXPERIMENT"]["PROPORTION"] * 0.1
GPG_OBJ.import_keys_file("../keys/NODE_"+str(id)+"/private.pem")
print("[NODE]Node at",portnum)

sio.connect('http://localhost:'+portnum,)
sio.wait()

