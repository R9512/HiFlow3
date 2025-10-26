import socketio,sys
import json
import pandas as pd
from time import sleep
from random import randint as rand
#from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor as LOF
from tomli import load

global validator
vsio=socketio.Client()
# @vsio.event
# def Write code here to let them start consensus 


class Validator:
	def __init__(self,fl_data,sio):
		"""
		fl_data is a dict which has weights, bias, accuracy,loss, with the round num as dict
		  Note: try not to call connect and wait in the constructor hehehe
		"""
		
		
		self.vsio = sio
		self.heroword=[]
		self.decision=0
		self.decided=False
		self.ultimate_decision=0
		self.weight_track =""
		with open("config.toml","rb") as f:
			config = load(f)
		self.n_neighbour = config["VALIDATOR"]["MIN_CLIENTS"]*config["VALIDATOR"]["MAX_ROUNDS"] //2
		self.k,self.ALPHA,self.BETA=config["VALIDATOR_UTIL"].values()

		self.create_model(fl_data)


	#=======Anamoly Detection Model==
	def create_model(self,data):
		"""
			****THIS IS NOT MODEL AGNOSTIC****
			Here we are doing only for Logistic Regression


			--- This thing is not correct
			Problem: data is a dict with weights and bias
			- It is a combination of list that is in the form of string, incase of mnist it is creating a problem
		"""

		data = pd.DataFrame(data)
		data.drop(["loss","accuracy","data_shape"],inplace=True)
		a = []
		b = []
		for x in range(data.shape[-1]):
			a.append(data.iloc[0].iloc[x])
			b.append(data.iloc[1].iloc[x])
		t_data1=pd.DataFrame(a)
		t_data2=pd.DataFrame(b) #Creating a dataframe for bias
		t_data1["bias"]=t_data2
		#All the modifications are done until here
		self.DATA=t_data1
		#Decision to be made here, whether should I consider the aggregated weights or all the weights given by 
		#nodes or just all the updates given by the nodes
		#self.MODEL = IsolationForest().fit(t_data1.values)

		self.MODEL = LOF(n_neighbors=1,novelty=True).fit(t_data1.values)

		#Data will be lost from this point as it may not be usefull

	
		
	#=======Snowball Protocol========

	def get_hero_opinion(self):
		data={"seed":455,"k":self.k,"weigths":"sairam"}
		data = json.dumps(data)
		self.vsio.emit("handle_hero_request",data)
		while len(self.heroword)!=self.k:
			pass

	def snowball_protocol(self):
		final_color,lastcolor,count =self.ultimate_decision,self.ultimate_decision,0
		counter=[0,0]
		f_counter=[0,0]
		self.decided=False
		
		r_count = 0
		print("="*25)
		print("Starting SnowBall Protocol")
		print("\tInitial Decision=>",self.ultimate_decision)
		while not self.decided:
			#print("\t\tStarting the round",r_count)
			r_count+=1
			self.vsio.sleep(0.5)	#To give breathing space to cordi? Try removing it and run agian
			self.get_hero_opinion()
			majority=False
			f_counter=self.heroword.count(0),self.heroword.count(1),
			max_c = max(f_counter)
			max_i = f_counter.index(max_c) 
			if(max_c >= self.ALPHA):
				majority=True
				if(max_i==lastcolor):
					count+=1
				else:
					lastcolor= max_i
					count = 0
			
				if(count >= self.BETA):
					self.decided=True
					final_color=max_i

		print("\tOut of Snowball.Final Decision=>",final_color)
		self.ultimate_decision=final_color
		
		self.vsio.emit("accepted_weights",str(self.ultimate_decision))
		








# if __name__=="__main__":
def main(weights):
	global validator
	validator = Validator(weights,vsio)

	print("Snow Ball Prorocol")
	# connected = False
	# sleep(0.5)
	# vsio.emit("submit_weights",json.dumps({"data":validator.DATA.to_dict()}))#Temp Fix
	# validator.snowball_protocol()
	# #vsio.wait()
	vsio.disconnect()
