import tkinter as tk                    
from tkinter import ttk
from tkinter import filedialog as fd
import toml
import pandas as pd

"""
Whatever input is givben add it to another dict
Build the final dict and check the conditions on top
of the new dict
"""
FILENAME=""

def execute():
  pass
def show_intro():
  #Intro should go here
  pass
def note_print(a,delete=False):
  #It will just append stuff to the last text box
  #The last textbox should be non editable+
  outro.config(state='normal')

  if(delete):
    outro.delete("1.0","end")

  outro.insert(tk.END,"\n - "+a)
  outro.config(state='disabled')
def toml_setter():
  global FILENAME
  data = toml.load("config.toml")
  #Check the conditions 
  cpubkey,cprikey=cpub_key.get(),cpri_key.get()
  conadd,target = con_add.get(),tar_var.get()
  numval,numpart = num_val.get(),num_part.get()
  vari,zet=var.get(),zeta.get()
  uniform = uni.get()

  vpubkey,vprikey=vpub_key.get(),vpri_key.get()
  vun,vva= vuni.get(),vvar.get()
  v_class,v_feat = n_class.get(),n_features.get()
  v_max,v_minc=max_rou.get(),min_cli.get()
  vbase = base.get()

  salpha,sbeta = alpha.get(),beta.get()
  sk=k.get()
  create = True
  #Check for integer 21
  change={'COORDINATOR':{},'VALIDATOR':{},'VALIDATOR_UTIL':{},'SETUP':{}}
  if(v_max != ""):
    try:
      v_max=int(v_max)
    except ValueError:
      note_print("Maximum Rounds in Validator is not set properly")
      create = False
    change["VALIDATOR"]['MAX_ROUNDS'] = v_max

  if(v_minc != ""):
    try:
      v_minc=int(v_max)
    except ValueError:
      note_print("Minimum Clients per Validator is not set properly")
      create = False
    change["VALIDATOR"]['MIN_CLIENTS'] = v_minc

  if(vbase != ""):
    try:
      vbase=int(vbase)
    except ValueError:
      note_print("Base in the Validator is not set properly")
      create = False
    change["VALIDATOR"]['BASE'] = vbase


  if(numval!=""):
    try:
      numval = int(numval)
    except ValueError:
      note_print("Number of Validators is not set properly")
      create = False
    change["COORDINATOR"]['NUM_VALIDATORS'] = numval

  if(numpart != ""):
    try:
      numpart = int(numpart)
    except ValueError:
      note_print("Number of Partitions is not set properly")
      create = False
    change["COORDINATOR"]["NUM_PARITIONS"] = numpart
  
  if(vari != ""):
    try:
      vari = int(vari)
    except ValueError:
      note_print("Varience in Coordinator is not set properly")
      create = False
    change["COORDINATOR"]["VARIENCE"] = vari

  if(zet != ""):
    try:
      zet = int(zet)
    except ValueError:
      note_print("Hyperparameter Zeta is not set properly")
      create = False
    change["COORDINATOR"]["ZETA"] = zet
  
  if(vva!=""):
    try:
      vva = int(vva)
    except ValueError:
      note_print("Varience in Validator is not set properly")
      create = False
    change["VALIDATOR"]["VARIANCE"]=vva

  if(v_class !=""):
    try:
      v_class = int(v_class)
    except ValueError:
      note_print("Validator Numb Feature ")
      create = False
    change["VALIDATOR"]["N_CLASSES"]=v_class
  
  if(v_feat !=""):
    try:
      v_feat = int(v_feat)
    except ValueError:
      note_print("Number of Features is not set properly")
      create = False
    change["VALIDATOR"]["N_FEATURES"] = v_feat


  if(vbase !=""):
    try:
      vbase = int(vbase)
    except ValueError:
      note_print("Base is not set properly")
      create = False
    change["VALIDATOR"]["BASE"] = vbase

  if(salpha !=""):
    try:
      salpha = int(salpha)
    except ValueError:
      note_print("Alpha Hyperparameter of Snowball is not set")
      create = False
    change["VALIDATOR_UTIL"]["ALPHA"] = salpha
  if(sbeta !=""):
    try:
      sbeta = int(sbeta)
    except:
      note_print("Beta hyperparameter of Snowball is not set")
      create = False
    change["VALIDATOR_UTIL"]["BETA"] = sbeta

  if(sk!=""):
    try:
      sk = int(sk)
    except:
      note_print(" K hyperparameter of Snowball is not set properly")
    change["VALIDATOR_UTIL"]["K"] = sk
  #Bool
  if(uniform != ""):
    try:
      uniform = bool(uniform)
    except ValueError:
      note_print("Uniform is not set")
    change["COORDINATOR"]["UNIFORM"] = uniform

  if(vun != ""):
    try:
      vun = bool(vun)
    except ValueError:
      note_print("Validator Uniform is not set")
    change["VALIDATOR"]["UNIFORM"] = vun

  #Check for the others
  #Len of pub=>42;pri=>66;con=42 private key 
  if(len(conadd)!=42 and conadd!=""):
    note_print("Contract address is not set properly.")
    create = False
  if(len(cpubkey)!=42 and cpubkey!=""):
    note_print("Public address of cordinator is not correct.")
    create = False
  if(len(vpubkey)!=42 and vpubkey!=""):
    note_print("Public address of validator is not correct.")
    create = False
  if(len(cprikey)!=66 and cprikey!=""):
    note_print("Coordinator Private key is not proper")
    create = False
  if(len(vprikey)!=66 and vprikey!=""):
    note_print("Validator private key is not proper")
    create = False

  if(target ==""):
    target = data["COORDINATOR"]["TARGET"]
  if(FILENAME ==""):
    FILENAME = data["COORDINATOR"]["PATH"]


  change["COORDINATOR"]["TARGET"] = target
  change["VALIDATOR"]["TARGET"] = target
  
  #It is Decided that SETUP is not necessary 
  #As this code itself will invoke the script
  #If not add the set up here
  #Order is :NUM_VALIDATORS = 8;MIN_CLIENTS = 2;PORT_NUM = 44600

  for x in change:
    for y in change[x]:
      data[x][y] =change[x][y] 
  #Creation of TOML config



  data1 = pd.read_csv(FILENAME)
  temp = data1.columns.str.contains(target)
  if(not (True in temp)):
    note_print("Target column is not found in the given CSV. It is case sensitive")
    create  =False
    return
  data["COORDINATOR"]["PATH"]=FILENAME
  if(data["VALIDATOR_UTIL"]["ALPHA"] < data["VALIDATOR_UTIL"]["K"]//2 ):
    note_print("Set the Alpha Hyperparameter > K/2")
    create  =False
    return
  if(not 2<data["VALIDATOR_UTIL"]["BETA"]<10):
    note_print("Beta value should be between 2 and 10")
    create  =False
    return
  if(not 1<data["VALIDATOR"]["MIN_CLIENTS"]<10):
    note_print("Minimum Nodes per Clients should be between 2 and 10")
    create  =False  
    return
  if(not 40000<= data["VALIDATOR"]["BASE"]<55000):
    note_print("Base Socket value should lie between 40000 and 55000. Ensure that 45045 is not used")
    create  =False  
    return


  if(create):
    with open('config.toml', 'w') as f:
      new_toml_string = toml.dump(data, f)
    note_print("Config is Valid . Please click Execute to start.",delete=True)
    note_print("[Note: If any field is left empty, the default value is taken]")
    note_print("\n\n\n\n\n\n\n\t\t Made with ❤ by Dept. of Mathematics and Computer Science, SSSIHL")
  else:
    note_print("*Please fix above mentioned problems*")
  
  #Just write into toml
  
def select_file():
  global FILENAME
  filetypes = (('text files', '*.csv'),)
  FILENAME = fd.askopenfilename(title='Open a file',initialdir='./',filetypes=filetypes)
    
    
  
root = tk.Tk()
root.title("Sairam")
root.geometry("800x500")
tabControl = ttk.Notebook(root)
  
tab1 = ttk.Frame(tabControl)#Starting Up Evrything
tab2 = ttk.Frame(tabControl)#For Coordinator
tab3 = ttk.Frame(tabControl)#For Validator
tab4 = ttk.Frame(tabControl)#For Valid _Util/Snowball
tab5 = ttk.Frame(tabControl)#For Script?
  
tabControl.add(tab1, text ='Setting Up')
tabControl.add(tab2, text ='Coordinator')
tabControl.add(tab3, text ='Validator')
tabControl.add(tab4, text ='Snowball')
tabControl.add(tab5, text ='Script')
tabControl.pack(expand = 1, fill ="both")

#Tab1 
ttk.Label(tab1, text="Introduction", font=('Helvetica bold', 20)).place(relx=0.5,rely=0.1,anchor='center')
intro = tk.Text(tab1,height = 23,width = 98,state='disabled')
intro.place(relx=0.0,rely=0.15)

#Tab2 Here
#Title
ttk.Label(tab2, text="Coordinator Details", font=('Helvetica bold', 20)).place(relx=0.5,rely=0.1,anchor='center')
ttk.Label(tab2, text="Coordinator Public Key:", font=('Helvetica bold', 12)).place(relx=0.0,rely=0.18)
cpub_key= ttk.Entry(tab2, width= 40)
cpub_key.place(relx=0.0,rely=0.23)

ttk.Label(tab2, text="Coordinator Private Key:", font=('Helvetica bold', 12)).place(relx=0.0,rely=0.35)
cpri_key= ttk.Entry(tab2, width= 40)
cpri_key.place(relx=0.0,rely=0.40)

ttk.Label(tab2, text="Contract Address:", font=('Helvetica bold', 12)).place(relx=0.0,rely=0.52)
con_add= ttk.Entry(tab2, width= 40)
con_add.place(relx=0.0,rely=0.57)

ttk.Label(tab2, text="Target Variable Name:", font=('Helvetica bold', 12)).place(relx=0.0,rely=0.69)
tar_var= ttk.Entry(tab2, width= 40)
tar_var.place(relx=0.0,rely=0.74)

open_button = ttk.Button(tab2,text='Select Dataset',command=select_file)
open_button.place(relx=0.0,rely=0.86)

#Second Column
ttk.Label(tab2, text="Number of Validators:", font=('Helvetica bold', 12)).place(relx=0.5,rely=0.18)
num_val= ttk.Entry(tab2, width= 40)
num_val.place(relx=0.5,rely=0.23)

ttk.Label(tab2, text="Number of Partitions:", font=('Helvetica bold', 12)).place(relx=0.5,rely=0.35)
num_part= ttk.Entry(tab2, width= 40)
num_part.place(relx=0.5,rely=0.40)

ttk.Label(tab2, text="Uniform:", font=('Helvetica bold', 12)).place(relx=0.5,rely=0.52)
uni= ttk.Entry(tab2, width= 40)
uni.place(relx=0.5,rely=0.57)

ttk.Label(tab2, text="Varience:", font=('Helvetica bold', 12)).place(relx=0.5,rely=0.69)
var= ttk.Entry(tab2, width= 40)
var.place(relx=0.5,rely=0.74)

ttk.Label(tab2, text="Zeta:", font=('Helvetica bold', 12)).place(relx=0.5,rely=0.86)
zeta= ttk.Entry(tab2, width= 40)
zeta.place(relx=0.5,rely=0.91)

#Tab3
ttk.Label(tab3, text="Validator Details", font=('Helvetica bold', 20)).place(relx=0.5,rely=0.1,anchor='center')
ttk.Label(tab3, text="Validator Public Key:", font=('Helvetica bold', 12)).place(relx=0.0,rely=0.18)
vpub_key= ttk.Entry(tab3, width= 40)
vpub_key.place(relx=0.0,rely=0.23)
 
ttk.Label(tab3, text="Validator Private Key:", font=('Helvetica bold', 12)).place(relx=0.0,rely=0.35)
vpri_key= ttk.Entry(tab3, width= 40)
vpri_key.place(relx=0.0,rely=0.40)

ttk.Label(tab3, text="Number of Classes:", font=('Helvetica bold', 12)).place(relx=0.0,rely=0.52)
n_class= ttk.Entry(tab3, width= 40)
n_class.place(relx=0.0,rely=0.57)

ttk.Label(tab3, text="Number of Features:", font=('Helvetica bold', 12)).place(relx=0.0,rely=0.69)
n_features= ttk.Entry(tab3, width= 40)
n_features.place(relx=0.0,rely=0.74)



#Second Column
ttk.Label(tab3, text="Max Round:", font=('Helvetica bold', 12)).place(relx=0.5,rely=0.18)
max_rou= ttk.Entry(tab3, width= 40)
max_rou.place(relx=0.5,rely=0.23)

ttk.Label(tab3, text="Minimum Clients per Validator:", font=('Helvetica bold', 12)).place(relx=0.5,rely=0.35)
min_cli= ttk.Entry(tab3, width= 40)
min_cli.place(relx=0.5,rely=0.40)

ttk.Label(tab3, text="Uniform:", font=('Helvetica bold', 12)).place(relx=0.5,rely=0.52)
vuni= ttk.Entry(tab3, width= 40)
vuni.place(relx=0.5,rely=0.57)

ttk.Label(tab3, text="Varience:", font=('Helvetica bold', 12)).place(relx=0.5,rely=0.69)
vvar= ttk.Entry(tab3, width= 40)
vvar.place(relx=0.5,rely=0.74)

ttk.Label(tab3, text="Base:", font=('Helvetica bold', 12)).place(relx=0.43,rely=0.86)
base= ttk.Entry(tab3, width= 40)
base.place(relx=0.3,rely=0.91)


#Tab4
ttk.Label(tab4, text="Snowball Details", font=('Helvetica bold', 20)).place(relx=0.5,rely=0.1,anchor='center')
ttk.Label(tab4, text="Beta:", font=('Helvetica bold', 12)).place(relx=0.5,rely=0.18)
ttk.Label(tab4, text="Alpha:", font=('Helvetica bold', 12)).place(relx=0.0,rely=0.18)
alpha= ttk.Entry(tab4, width= 40)
alpha.place(relx=0.0,rely=0.23)
ttk.Label(tab4, text="Alpha: Represents the number responses to considered as maximum", font=('Helvetica bold', 12)).place(relx=0,rely=0.50)
ttk.Label(tab4, text="Beta: Represents the number of consecutive responses to get", font=('Helvetica bold', 12)).place(relx=0,rely=0.60)
ttk.Label(tab4, text="K: Number of samples to be asked", font=('Helvetica bold', 12)).place(relx=0,rely=0.7)
ttk.Label(tab4, text="Alpha should be grater than half value of K",foreground="blue", font=('Helvetica bold', 12)).place(relx=0,rely=0.8)
ttk.Label(tab4, text="Higher Beta value leads to more time to converge, smaller the beta less the trust",foreground="blue" ,font=('Helvetica bold', 12)).place(relx=0,rely=0.9)
beta= ttk.Entry(tab4, width= 40)
beta.place(relx=0.5,rely=0.23)

ttk.Label(tab4, text="K:", font=('Helvetica bold', 12)).place(relx=0.4,rely=0.35)
k= ttk.Entry(tab4, width= 40)
k.place(relx=0.3,rely=0.40)
#Tab5
ttk.Button(tab5,text='Create Config',command=toml_setter).place(relx=0.2,rely=0.05)
ttk.Button(tab5,text='Execute',command=execute).place(relx=0.6,rely=0.05)
outro = tk.Text(tab5,height = 23,width = 98,state='disabled')
outro.place(relx=0.0,rely=0.15)



if __name__ == "__main__":
  root.mainloop()  