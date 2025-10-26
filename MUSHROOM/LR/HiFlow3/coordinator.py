import os

import eventlet 
import socketio
import json
import numpy as np
import time
import pickle
import pandas as pd
import pandas as pd
import json
import gnupg
import warnings,sys,toml
import ipfs_api as ips
from web3 import Web3
from sklearn.linear_model import LogisticRegression
from random import randint as rand
from time import sleep
from tomli import load
from shutil import rmtree
from os import mkdir
from shutil import rmtree



GPG_OBJ = gnupg.GPG(gnupghome='/home/swami/.gnupg/')
GPG_OBJ.import_keys_file("../keys/DECENTRALIZE_AGGREGATOR/private.pem")
GPG_OBJ.encoding = 'utf-8'
sio = socketio.Server(logger=False,engineio_logger=False)
app = socketio.WSGIApp(sio)
sock = eventlet.listen(('', 45045))
CONNECTED=[]
RECORD={}

WEIGHTS=[]#HOLDS THE CID
WEIGHTS1=[]#HOLDS THE ACTUAL WEIGHTS
ACCEPTED_WEIGHTS={}
ORIGINAL_DECISION=[]

RECIEVED_COUNTER=0
CURRENT_WEIGHT=0
NUM_VALIDATORS=5
DECISION=0#use in the start_weight_process
WROTE_TO_BC_COUNTER = 0
GAS_USED = 0
VALIDATOR_GAS_USED={}
def start_weight_process(n):#Send Only last weights
	print("=="*10)
	print("\t\tProcessing the Weights",n)
	print("Current _WEIGHTS,",WEIGHTS[n])
	print("CALLING THE TAKE WEITHS with CURRENT WEIGHTS",n)
	sio.emit("take_weights",{"data":WEIGHTS[n],"round":n},room="validators")
	sio.sleep(3)
	print("\t\tStarting Snowball")
	sio.emit("start_snowball","Lets Go Bouys",room="validators")

def trust_cal(original,given):
	score = 0
	print("\t\t Finding For:",given,"Original",original)
	given=given[1:]
	for i,x in enumerate(given):
		#print("Process",x)
		if(given[i]==original[i]):
			#print("\t\t\t\t Adding 10")
			score+=10
		else:
			#print("\t\t\t\t Removing 10")
			score-=10
	if score <1:
		score=1
	return score

def zeta_handler():
    """
    Handles zeta
    Needs the location of JSON
    """
    sio.sleep(2)
    global CON_ADD,CONN,CONTRACT,NUM_VALIDATORS,ORIGINAL_DECISION,ZETA
    
    ORIGINAL = ""
    
    for x in ORIGINAL_DECISION:
    	ORIGINAL+=str(x)
    #print(ORIGINAL,str(CONTRACT.functions.get_values(str("0")).call()))
    TRUST = [trust_cal(ORIGINAL,str(CONTRACT.functions.get_values(str(x)).call())) for x in range(NUM_VALIDATORS)]
    temp_total = sum(TRUST)
    print("\t\tTrust:",TRUST)
    TRUST=[round(x/temp_total,4) for x in TRUST]
    portion = 1/NUM_VALIDATORS
    remaining = 1
    FINAL_WEIGHTAGE = [0 for x in range(NUM_VALIDATORS)]
    
    for i,x in enumerate(ORIGINAL):
        if x=="1":
            FINAL_WEIGHTAGE[i]=portion
            remaining-=portion
    TRUST=[x*remaining for x in TRUST]
    for i in range(0,NUM_VALIDATORS):
        FINAL_WEIGHTAGE[i]+=TRUST[i]
        FINAL_WEIGHTAGE[i]=round(FINAL_WEIGHTAGE[i],4)
    return FINAL_WEIGHTAGE

def final_model_generator():
	global GENESIS_MODEL_CID,GPG_OBJ,ZETA
	print("The Following is:\n",ACCEPTED_WEIGHTS)
	using_zeta = False
	if(len(ACCEPTED_WEIGHTS) == 0 or len(ACCEPTED_WEIGHTS) < ZETA):
		print("Generating Using ZETA...")
		weightage = zeta_handler()
		print("Final Model Weightage:",weightage)
		data = pd.DataFrame.from_dict(WEIGHTS1)
		data=data.mul(pd.Series(weightage),axis=0)
		using_zeta = True
		
	else:
		weights = [ json.loads(ACCEPTED_WEIGHTS[x]) for x in ACCEPTED_WEIGHTS]
		data = pd.DataFrame.from_dict(weights)
	
	avg_weights=data.mean(axis=0).tolist()
	weights,bias = avg_weights[:-1],avg_weights[-1]
	#Reading From IPFS
	enc_gen_model = ips.read(GENESIS_MODEL_CID) 
	decrypted_data = GPG_OBJ.decrypt(enc_gen_model)
	g_model = pickle.loads(decrypted_data.data)

	GENESIS_DATA = g_model.coef_.tolist()[0]
	for x in range(len(GENESIS_DATA)):#Adding the Model Updates to Genisis Model to Create Final Model
		weights[x]+=GENESIS_DATA[x]
		

	model1 = LogisticRegression(penalty="l2",max_iter=1,warm_start=True,)
	model1.classes_ = np.array([i for i in range(2)])
	model1.coef_ = np.array([weights])#Should Add it with genesis model
	if model1.fit_intercept:#Should Add with Genesis Model as Well
	    model1.intercept_ = bias+g_model.intercept_[0]
	#Savin the model
	model_name =str(time.ctime(time.time())).replace(" ","")[3:-4]
	model_name=model_name[0:4]+"_"+model_name[4:]
	model_name+="_HiFloW3.pickle"
	if(using_zeta):
		model_name = "Z"+model_name
	file = open(model_name,"wb")
	pickle.dump(model1,file)
	file.close()
	with open("signal.flag", 'w') as file:
		pass
	#Remove the Genis Herhe
	return("Model Is generated")

@sio.event
def connect(sid, environ):
    print("A Validator Tried Connection")

@sio.event
def ready(sid):
	CONNECTED.append(sid)
	sio.enter_room(sid,"validators")
	print("Connected",len(CONNECTED))

@sio.event
def disconnect(sid):
	sio.leave_room(sid,"validators")
	CONNECTED.remove(sid)
	if(len(CONNECTED)==0):
		sio.emit("cood_disconnect","Sairam Model is Generated",room="validators")
		sio.close_room("validators")
		print("Process Ended")
		#rmtree("data/genesis_model.pkl")
		#sock.close()=>Uncomment This After FUlly Dev

@sio.event
def handle_hero_request(sid,data):
	data=json.loads(data)
	k=data.pop("k")
	data["for"]=sid
	RECORD[sid]=[data["seed"],k,[]]
	
	#=====Selecting Valid Heros
	#	Assumption: k<len(Connected)-1
	track=[False for x in CONNECTED]
	count,choice=0,0
	while count<k:
		choice=rand(0,len(CONNECTED)-1)
		if(track[choice] or CONNECTED[choice] == sid):
			continue
		track[choice] = True
		count+=1
		sio.emit("give_opinion",json.dumps(data),room=CONNECTED[choice])


@sio.event
def opinion_handler(sid,data):
	data=json.loads(data)
	if( RECORD.get(data["for"],9512) != 9512):
		k,current = RECORD[data["for"]][1],RECORD[data["for"]][-1]
		current.append(data["opinion"])
		if(len(current)==k):
			sio.emit("take_hero_word",json.dumps({"opinion":current}),room=data["for"])
			RECORD.pop(data["for"])



@sio.event
def submit_weights(sid,data):
	global GPG_OBJ
	cid = data
	print("\t\t\t Got The Weights @",cid)
	data=ips.read(data)
	decrypted_data = GPG_OBJ.decrypt(data.decode())
	payload=json.loads(decrypted_data.data.decode())
	WEIGHTS1.append(payload)
	WEIGHTS.append(cid)
	
	if(len(WEIGHTS)==NUM_VALIDATORS):#All Weights Are recieved Start Broasdcasting
		start_weight_process(0)

@sio.event
def accepted_weights(sid,data):

	global RECIEVED_COUNTER,CURRENT_WEIGHT,ACCEPTED_WEIGHTS
	global WROTE_TO_BC_COUNTER, NUM_VALIDATORS
	global GPG_OBJ,GAS_USED,VALIDATOR_GAS_USED
	RECIEVED_COUNTER+=1
	
	if(data=="1" and ACCEPTED_WEIGHTS.get(CURRENT_WEIGHT,-1) == -1):
		print("\t\tAdding to ACEPTE")#Hadnle Duplicated
		ORIGINAL_DECISION[CURRENT_WEIGHT] = 1#Just keepng track of the decision process made at cood used in Zeta
		data=ips.read(WEIGHTS[CURRENT_WEIGHT])
		decrypted_data = GPG_OBJ.decrypt(data.decode())
		ACCEPTED_WEIGHTS[CURRENT_WEIGHT]=decrypted_data.data.decode()


	if(RECIEVED_COUNTER==len(CONNECTED) and CURRENT_WEIGHT <= len(WEIGHTS)-1):
		RECIEVED_COUNTER=0
		CURRENT_WEIGHT+=1

		if CURRENT_WEIGHT != len(WEIGHTS):
			print("\t\tStarting Next Round")
			start_weight_process(CURRENT_WEIGHT)
		else:
			print(" Preparing to Generate Final Mode")
			sio.sleep(5)
			sio.emit("save_opinion",room="validators")
			print(time.time())
			sio.sleep(5)
			print(time.time())
			print("If it is zero then it is a problem",WROTE_TO_BC_COUNTER)
			WROTE_TO_BC_COUNTER = 0
			print(final_model_generator())
			

			sio.emit("cood_disconnect","Sairam Model is Generated",room="validators")
			sio.close_room("validators")
			with open("done.flag","wb") as f:
				f.write(b"done")
			with open("Coordinator.txt", "a") as myfile:
				myfile.write(str(GAS_USED)+",")
			sio.shutdown()

@sio.event
def wrote_to_BC(sid,data):
	global WROTE_TO_BC_COUNTER,VALIDATOR_GAS_USED
	VALIDATOR_GAS_USED[data["NUMBER"]] = data["GAS"]
	print("\t\t\t Got the SIGNAL",data,data["GAS"])
	WROTE_TO_BC_COUNTER+=1


def initialize_parameter():
	global C_PRI,C_PUB,CON_ADD,NUM_VALIDATORS,NUM_PARITIONS,PATH,TARGET,UNIFORM,VARIENCE,ZETA	
	global CONN,CONTRACT
	global ORIGINAL_DECISION, GAS_USED
	with open("config.toml","rb") as f:
		config = load(f)
	C_PRI,C_PUB,CON_ADD,NUM_VALIDATORS,NUM_PARITIONS,PATH,TARGET,UNIFORM,VARIENCE,ZETA,_= config["COORDINATOR"].values()#_ is temp fix
	ORIGINAL_DECISION=[0 for x in range(NUM_VALIDATORS)]
	with open("./SmartContract.json") as f:
		info_json = json.load(f)

	abi = info_json["abi"]
	print("This is the problem")
	try:
		CONN = Web3(Web3.HTTPProvider("HTTP://127.0.0.1:8545"))
	except ConnectionError  as e:
		print("Please set up Web3 Stuff")
	CONN.eth.default_account = CONN.eth.accounts[0]
	CONTRACT = CONN.eth.contract(address=CON_ADD,abi=abi)
	tra1 = CONTRACT.functions.set_up(NUM_VALIDATORS).transact()
	tx_receipt = CONN.eth.wait_for_transaction_receipt(tra1)
	GAS_USED += tx_receipt["gasUsed"]
	print("Smart Contract is ready")


def data_prep():
	global NUM_VALIDATORS,NUM_PARITIONS,PATH,TARGET,UNIFORM,VARIENCE

	data = pd.read_csv(PATH)
	data=data.drop_duplicates(keep=False)
	data=data.sample(frac=1)
	data_len = data.shape[0]
	mini = data_len//(NUM_PARITIONS)
	maxi = mini+VARIENCE
	PARTITIONS=[]
	while (data.shape[0]>maxi) :

		if not UNIFORM and data.shape[0]>maxi:
			maxi=mini+rand(10,VARIENCE)
			part = data.sample(n=rand(mini-VARIENCE,maxi))

		elif(data_len//(NUM_PARITIONS) < data.shape[0]):
			maxi=data_len//(NUM_PARITIONS)
			part = data.sample(n=data_len//(NUM_PARITIONS))
		temp = pd.concat([part,data])
		data=temp.drop_duplicates(keep=False)
		PARTITIONS.append(part)

	if not UNIFORM:

		if(data.shape[0]>mini):
			PARTITIONS.append(data[:mini])
		while len(PARTITIONS)>NUM_PARITIONS:
			PARTITIONS.pop()
	else:
		while len(PARTITIONS) != NUM_PARITIONS:
			PARTITIONS.append(data[:mini])
			data=data[:mini]
	rmtree("data",ignore_errors=True)
	mkdir("data")
	with open("config.toml","rb") as f:
		config = load(f)
	FLIP,PROPORTION = config["EXPERIMENT"]["FLIP_NUM"],config["EXPERIMENT"]["PROPORTION"]
	print("Got the flip and Proportion:",FLIP,PROPORTION)
	for x in range(0,FLIP):
		col = config["COORDINATOR"]["TARGET"]
		data=PARTITIONS[x]
		print("Before",data[col].value_counts())
		df_1 = data[data[col]==1].reset_index()
		df_0 = data[data[col]==0].reset_index()
		fraction_0,fraction_1 =int(df_0.shape[0]*PROPORTION),int(df_1.shape[0]*PROPORTION)
		df_1.loc[1:fraction_1,col] = 0
		df_0.loc[1:fraction_0,col] = 1
		data_frame = pd.concat([df_1,df_0])
		data_frame.pop("index")
		print("\nAfter",data_frame[col].value_counts())
		PARTITIONS[x] = data_frame
	for y,x in enumerate(PARTITIONS):
	 	x.to_csv("data/"+str(y)+".csv",index=False)

def create_genisis_model():
	global GENESIS_MODEL_CID,GENESIS_MODEL_WEIGHT,NUM_PARITIONS,TARGET,GPG_OBJ
	data = pd.read_csv("./data/"+str(NUM_PARITIONS-1)+".csv")
	y = data.pop(TARGET)
	genesis_model = LogisticRegression(max_iter=2,warm_start=True)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		genesis_model.fit(data.values,y)
	temp = pickle.dumps(genesis_model)
	with open("config.toml","rb") as f:
		data = load(f)
	num_valid = data["COORDINATOR"]["NUM_VALIDATORS"]
	recipients = []
	for x in range(num_valid):
		recipients.append("VALIDATOR_"+str(x+1)+"@HiFlow3.com")

	encrypted_data = GPG_OBJ.encrypt(temp,recipients,sign="DECENTRALIZE_AGGREGATOR@HiFlow3.com")
	with open("e_geneisis_model.pkl", "wb") as f:
		f.write(encrypted_data.data)
	GENESIS_MODEL_CID = ips.publish("e_geneisis_model.pkl")

	data["COORDINATOR"]["GENISIS_MODEL_CID"] = GENESIS_MODEL_CID
	with open("config.toml", 'w') as f:
            new_toml_string = toml.dump(data, f)
	
	print("File is Uploaded to IPFS: CID:",GENESIS_MODEL_CID)
	print("Updated the Config File")
	os.remove("e_geneisis_model.pkl")
	print("Deleted The FIle")

	

initialize_parameter()
create_genisis_model()

#sleep(5)
print("Yeah ready")
eventlet.wsgi.server(sock, app,log=None,log_output=False)

