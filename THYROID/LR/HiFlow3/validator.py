import os
# os.environ['EVENTLET_NO_GREENDNS'] = 'yes'
# import eventlet 
import warnings
import socketio
import pandas as pd
import json,sys
import numpy as np
import validator_util as utils
import pickle 
import gnupg
import ipfs_api as ips
import eventlet
from time import sleep
from datetime import datetime as dt
from web3 import Web3
from random import randint as rand
from tomli import load
from sklearn.model_selection import train_test_split as split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss as loss
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import roc_auc_score as ras
from os import remove

sio = socketio.Server(logger=False,engineio_logger=False)
app = socketio.WSGIApp(sio)
GPG_OBJ = gnupg.GPG(gnupghome='/home/swami/.gnupg/')

GPG_OBJ.encoding = 'utf-8'
PARTITIONS=[]
CONNECTED_NODES=0
NODES={}	#A dict with keys of the sid
MODELS={}	#round_num=>weights,bias,loss,accuracy,data 
CENTRAL_MODEL={}#Holds the model
TEST_MODEL = 0
NOVELTY_OPINION = {}#HOLDS THE OPINION ALONG WITH ROUND NUMBER
global MIN_CLIENTS,MAX_ROUNDS,PARTITION_NUM,TARGET,N_CLASSES,N_FEATURES
GAS_USED = 0
#==================================
def prepare_data(data_part):
	"""
		- Idea is to give the deired number of paritions as requested
			- If uniform is set to True, all the partions will have same number of records
			- Else they will have different values

		* Pass the target table to the nodes so that they will do the test train split
		
		- Something conk is hapenning with 6 other than that all is fine mostly
		- Limit to number of node to 10 for each validator
		

		#Need to Deal with the error
	"""
	global BASE,MAX_ROUNDS,MIN_CLIENTS,N_CLASSES,N_FEATURES,TARGET,UNIFORM,VARIANCE
	if( not os.path.exists("./data/"+data_part+".csv")):
		data_part+="_m"


	data = pd.read_csv("./data/"+data_part+".csv")#CHANGEHERE
	partitions = MIN_CLIENTS
	uniform = UNIFORM
	#N_CLASSES,N_FEATURES=data[TARGET].nunique(),data.shape[-1]-1

	data=data.drop_duplicates(keep=False)
	data=data.sample(frac=1)
	data_len = data.shape[0]
	mini = data_len//(partitions)
	maxi = mini+rand(0,50)
	while (data.shape[0]>maxi) :
		maxi = mini+rand(0,50)
		if not uniform and data.shape[0]>maxi:
			part = data.sample(n=rand(mini,maxi))
		elif(data_len//(partitions) < data.shape[0]):
			MAX=data_len//(partitions)
			part = data.sample(n=data_len//(partitions))
		temp = pd.concat([part,data])
		data=temp.drop_duplicates(keep=False)
		PARTITIONS.append(part)
	
	if not uniform:

		if(data.shape[0]>mini):
			PARTITIONS.append(data[:mini])
		while len(PARTITIONS)>partitions:
			PARTITIONS.pop()
	else:
		while len(PARTITIONS) != partitions:
			PARTITIONS.append(data[:mini])
			data=data[:mini]


def send_data():
	"""
	Sends data to the respective Nodes
	Payload Structure:
		- "data": Has The Data
		- "num_data_records": Number of data records (Is it necessary?)
		- "target" : Target Column Name
		- "n_classes": Number of classes
		- "n_features":Number of features
		- "round_num": Currently Happening Round
	"""
	global BASE,GENESIS_MODEL,N_CLASSES,N_FEATURES,NUMBER,OLD_DATA,SAVE_PART,TARGET
	global PORT_NUM
	#Genisis_Model  Will Be Read Here
	#Send The Weights 
	model = GENESIS_MODEL
	if(SAVE_PART):
		number = NUMBER
	data_sent = 0
	for i,x in enumerate(NODES.keys()):
		payload={"data":PARTITIONS[i],"num_data_records":PARTITIONS[i].shape[0],"target":TARGET,
				"n_classes":N_CLASSES,"n_features":N_FEATURES,"round_num":1}
		NODES[x]=payload
		NODES[x]["STATUS"]=["SENDING THE DATA"]
		print("[V"+str(NUMBER)+"]Data that is sent is:",PARTITIONS[i].shape)
		payload["genesis_model"] = {"weights":model.coef_.tolist(),"bias":model.intercept_.tolist(),}
		OLD_DATA={"weights":payload["genesis_model"]["weights"][0],"bias":payload["genesis_model"]["bias"][0]}
		payload["data"]=PARTITIONS[i].to_json()
		payload=json.dumps(payload)
		size_bytes = len(payload.encode('utf-8'))
		data_sent+=size_bytes
		sio.emit("recieve_data",payload,room=x)
	with open('network.txt', 'a') as f:
		f.write(str(PORT_NUM)+","+str(data_sent)+"\n")
		#PARTITIONS[i].to_csv("./mnist/9/n"+str(number)+"_"+str(i)+".csv",index=False)#CHANGEHERE



def federated_average(roundnum):
	
	global OLD_DATA
	weights = {}

	for x in NODES:
		weights[x]=NODES[x][roundnum]["weights"]
	
	fed_weight,fed_bias = [],[]
	new_model=[]
	weight_avg = [NODES[x][roundnum]["num_data_records"] for x in NODES]

	total=sum(weight_avg)

	weight_avg=[x/total for x in weight_avg]

	count = 0

	for i,x in enumerate(weights):
		count+=1
		rows,column = len(weights),len(weights[x]["weights"])
	
		impact = weight_avg[i]
		current = weights[x]["weights"]
		fed_weight.append([z*impact for z in current])
		fed_bias.append(impact*weights[x]["bias"][0])
	#sOMETHING IS GOINGON HERE, I REMEBER OLD DATA  CONTAINS THE NEW DATA
	for x in range(column):
		c_total = 0
		for y in range(rows):
			c_total+=fed_weight[y][x]
		new_model.append(c_total)

	for x in range(len(OLD_DATA["weights"])):
		OLD_DATA["weights"][x]+=new_model[x]
	OLD_DATA["bias"]+=sum(fed_bias)
	MODELS[roundnum]={"weights":new_model,"bias":sum(fed_bias)}
	# print("+++++++++++++++"*10)
	# print("WEIGHTAGE",weight_avg)
	# print("AVERAGEED IS:",new_model)
	# print("+++++++++++++++"*10)


def model_tester(round_num):
	"""
		- This is only for the Logistic Regression 
		- The built platform is NOT model agnostic
	"""
	global N_CLASSES,N_FEATURES,TEST_MODEL

	if TEST_MODEL == 0:	
		TEST_MODEL = LogisticRegression(penalty="l2",max_iter=1,warm_start=True,class_weight={1:0.542,0:6.464})
		TEST_MODEL.classes_ = np.array([i for i in range(N_CLASSES)])
		TEST_MODEL.coef_ = np.zeros((N_CLASSES,N_FEATURES))

	weights,bias = MODELS[round_num]["weights"],MODELS[round_num]["bias"]
	
	TEST_MODEL.coef_ = np.array([weights])
	if TEST_MODEL.fit_intercept:
	    TEST_MODEL.intercept_ = np.array([bias])

	data = PARTITIONS[-1].copy()
	y= data.pop(TARGET)
	data,y=data.values,y.values

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		y_hat = TEST_MODEL.predict(data)
	
	lgl= loss(y,y_hat)
	accuracy = acc(y,y_hat)
	print("[V"+str(NUMBER)+"]Round Num,Loss,Accuracy,AUC",round_num,lgl,accuracy,ras(y,y_hat))
	MODELS[round_num]["loss"]=lgl
	MODELS[round_num]["accuracy"]=accuracy
	MODELS[round_num]["data_shape"]=data.shape
	

def passing_on():
	"""
		well passes on the datato the valudator utils
		Things to be done:
			- reshape the data weights and bias to single shape
			- train the model
	"""

	global GENESIS_MODEL,OLD_DATA
	vsio=socketio.Client() #Cleint for the coordinator
	validator = utils.Validator(MODELS,vsio)
	vsio.connect("http://localhost:45045")
	while not vsio.connected:
	        pass
	print("[V]Connected to Coordinator,Just Now. Emiting the ready signal")
	vsio.emit("ready")




	@vsio.event
	def take_weights(data):
		global GPG_OBJ,NUMBER,NOVELTY_OPINION
		print("PROCESSING THE WEIGHTS:\n",data)
		current_round = data["round"]
		data = data["data"]
		data=ips.read(data)
		decrypted_data = GPG_OBJ.decrypt(data.decode())
		data=json.loads(decrypted_data.data.decode())
		#here some problem is therE????
		
		t=validator.MODEL.predict([data])[0]
		validator.ultimate_decision=1
		if(t==-1):
			validator.ultimate_decision=0
		
		NOVELTY_OPINION[current_round] = validator.ultimate_decision
		print("[V"+str(NUMBER)+"]\t\tPrediction from Model:",validator.ultimate_decision)
		validator.weight_track+=str(validator.ultimate_decision)

	@vsio.event
	def start_snowball(data):
		validator.snowball_protocol()
		
		
	@vsio.event
	def give_opinion(data):
		data=json.loads(data)
		data["opinion"]=validator.ultimate_decision
		#data["opinion"]=1 for with out validation
		data.pop("weigths")
		vsio.emit('opinion_handler',json.dumps(data))
		#Send it to someone

	@vsio.event
	def take_hero_word(data):
		data=json.loads(data)
		data=data["opinion"]
		validator.heroword=data.copy()


	@vsio.event
	def cood_disconnect(data):
		print("[V]Message From Co:",data)
		vsio.disconnect()
		vsio.shutdown()
	
	@vsio.event
	def save_opinion():
		global CONTRACT,CONN,NUMBER,V_PUB,V_PRI
		global NOVELTY_OPINION,GAS_USED
		print("HERE IS MY LOF OPINION",NOVELTY_OPINION)
		print("&&&&&"*5)
		print("From Save Opinion",validator.weight_track)
		NONCE = CONN.eth.get_transaction_count(V_PUB)+NUMBER
		print("NONCE",NONCE)
		print("[V]About to start transaction")
		sio.sleep(NUMBER)
		transaction = {'chainId': 31337,"nonce":NONCE}
		#validator.weight_track = "12010" ONLY FOR TESTING
		print("From Save Opinion",validator.weight_track)
		print("Writing To BC:",int("1"+validator.weight_track))
		tra1 = CONTRACT.functions.insert_record(str(NUMBER),int("1"+validator.weight_track)).build_transaction(transaction)
		signed_tx1 = CONN.eth.account.sign_transaction(tra1,private_key=V_PRI)
		tra1_hash = CONN.eth.send_raw_transaction(signed_tx1.raw_transaction)
		reci = CONN.eth.wait_for_transaction_receipt(tra1_hash)
		GAS_USED += reci["gasUsed"]
		print("[V]",str(NUMBER),"Wrote into Blockchain",validator.weight_track)
		sio.emit("wrote_to_BC",{"NUMBER":NUMBER,"GAS":GAS_USED})
		with open(str(NUMBER)+".txt", "a") as myfile:
			myfile.write(str(GAS_USED)+";")



	GENESIS_DATA = GENESIS_MODEL.coef_.tolist()[0]
	t = []
	#Remeber OLD_DATA is the last new one
	for x in range(0,len(GENESIS_DATA)):#Finding the updates
		t.append(OLD_DATA["weights"][x] - GENESIS_DATA[x])
	t.append(OLD_DATA["bias"]-GENESIS_MODEL.intercept_[0])
	#print("These are my Weights:",t)
	payload_enc = json.dumps(t)
	encrypted_data = GPG_OBJ.encrypt(payload_enc,"DECENTRALIZE_AGGREGATOR@HiFlow3.com")
	name = str(rand(0,5000))+"_"+str(dt.now())+".txt"
	name = "./data/"+name
	with open(name,"w") as file:
		file.write(encrypted_data.data.decode())
	cid = ips.publish(name)
	print("My WEIGHTS CID:",cid)
	remove(name)
	vsio.emit("submit_weights",cid)
	vsio.wait()	

#==================================
@sio.event
def connect(sid, environ):
	print("[V]A Node tried to connect")
 
@sio.event
def ready(sid):  
	print("GOT READY")
	global CONNECTED_NODES
	sio.enter_room(sid, 'nodes')
	NODES[sid]=[]
	CONNECTED_NODES+=1
	if MIN_CLIENTS==CONNECTED_NODES :
		send_data()
	else:
		print("[V]\t Node Connected! Waiting for",MIN_CLIENTS-(len(NODES))," more clients") 

@sio.event
def disconnect(sid):
	global CONNECTED_NODES

	CONNECTED_NODES-=1
	NODES.pop(sid)

@sio.event
def status_giver(sid,message):
	NODES[sid]["STATUS"].append(message)


@sio.event
def update_round(sid,message):
	global GPG_OBJ,OLD_DATA
	fed_avg_ready= True
	data=ips.read(message)
	decrypted_data = GPG_OBJ.decrypt(data.decode())
	payload=json.loads(decrypted_data.data.decode())
	temp_rn = "round"+str(payload["round_num"])
	
	NODES[sid][temp_rn] = payload
	for x in NODES:
		
		if(NODES[x].get(temp_rn,-1) == -1):
			fed_avg_ready=False
	
	
	if(fed_avg_ready):
		if(payload["round_num"] < MAX_ROUNDS+1):
			federated_average(temp_rn)#Change This As Well, USe the OLD MODEL;Save the OLD MODEL AS WELL
			temp = {"weights":OLD_DATA["weights"],"bias":OLD_DATA["bias"],"round_num":payload["round_num"]+1}
			sio.emit("fl_round",temp,room="nodes")
			model_tester(temp_rn)
		else:

			sio.emit("on_disconnect","Federated Learning is Done",room="nodes")
			sio.close_room("nodes")
			passing_on()

			


def initialize_parameter(portnum):
	global BASE,CON_ADD,MAX_ROUNDS,MIN_CLIENTS,N_CLASSES,N_FEATURES,SAVE_PART,TARGET,UNIFORM,VARIANCE,V_PRI,V_PUB	
	global CONN,CONTRACT,NUMBER
	global GENESIS_MODEL,GAS_USED
	global GPG_OBJ
	with open("config.toml","rb") as f:
		config = load(f)

	BASE,CON_ADD,MAX_ROUNDS,MIN_CLIENTS,N_CLASSES,N_FEATURES,SAVE_PART,TARGET,UNIFORM,VARIANCE,V_PRI,V_PUB = config["VALIDATOR"].values()
	cid = tuple(config["COORDINATOR"].values())[-1]
	print("READING:",cid)
	enc_gen_model = ips.read(cid) 
	#print(enc_gen_model)
	decrypted_data = GPG_OBJ.decrypt(enc_gen_model)
	GENESIS_MODEL = pickle.loads(decrypted_data.data)	
	with open("./SmartContract.json") as f:
		info_json = json.load(f)

	abi = info_json["abi"]
	try:
		CONN = Web3(Web3.HTTPProvider("HTTP://127.0.0.1:8545"))
	except OSError as e:
		print("[V]Web3 is not settedup")

	CONN.eth.default_account = CONN.eth.accounts[1]
	CONTRACT = CONN.eth.contract(address=CON_ADD,abi=abi)
	#Becoming Valid Validator
	nonce = CONN.eth.get_transaction_count(V_PUB) + 0#(44601 - portnum) #Unique Nonce
	print("THGIS IS THE NOCNCE:",nonce)
	t1_txn = CONTRACT.functions.auth_fund().build_transaction({
    'chainId': 31337,
    'gas': 70000,
    'maxFeePerGas': CONN.to_wei('2', 'gwei'),
    'maxPriorityFeePerGas': CONN.to_wei('1', 'gwei'),
    'nonce': nonce,
})
	signed_txn = CONN.eth.account.sign_transaction(t1_txn, private_key=V_PRI)
	tra1 = CONN.eth.send_raw_transaction(signed_txn.raw_transaction)
	tx_receipt = CONN.eth.wait_for_transaction_receipt(tra1)
	GAS_USED += tx_receipt["gasUsed"]
	print("[V]Smart Contract is ready")
	NUMBER = portnum -BASE

	name = str(NUMBER)
	if(config["EXPERIMENT"]["MALICIOUS"][NUMBER] == 1):
		name+="_m"
	return name


##MAIN
global PORT_NUM
portnum,id=int(sys.argv[1]),int(sys.argv[2])
PORT_NUM = portnum
GPG_OBJ.import_keys_file("../keys/VALIDATOR_"+str(id)+"/private.pem")
data_part=initialize_parameter(portnum)
prepare_data(str(data_part))

print("[V]A validator started @",str(portnum))
eventlet.wsgi.server(eventlet.listen(('', portnum)), app,log=None,log_output=False)
 
