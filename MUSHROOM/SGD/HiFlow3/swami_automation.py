#Automation
"""
    Things to done:
        - Change the FLIPNUM and PROPORTION in the config file
        - Run the model once
        - CSV files will be generated. Move them to "csv" directory under sub dir as x_y
            where x=FLIPNUM,y=PROPORTION
            - rename these as 0....n
        - Move the model to "model" dir with the same naming scheme
"""
import toml
import os
from datetime import datetime
from time import sleep
from shutil import move

def pickle_counter():
    count = 0
    for z in os.listdir("./"):
        if(".pickle" in z ):
            count+=1
    return count

ITER=10
CONFIG_PATH = "./config.toml"
zeta = 0
for zeta in range(0,6):
    current = ""
    os.mkdir("./models_zeta="+str(zeta))
    for flipnum in range(0,4):
        for x in range(0,9,2):
            current = str(flipnum)+","+str(x)
            proportion =x/10
            data = toml.load(CONFIG_PATH)
            data["EXPERIMENT"]["FLIP_NUM"],data["EXPERIMENT"]["PROPORTION"] = flipnum,proportion
            data["COORDINATOR"]["ZETA"] = zeta
            with open(CONFIG_PATH, 'w') as f:
                new_toml_string = toml.dump(data, f)
            while pickle_counter() < ITER:
                os.system("./try.sh")
                sleep(5)
            #Add codesuch a way that it does deletesmodels and csv
            newpath = "./models_zeta="+str(zeta)+"/"+str(current)
            os.mkdir(newpath)
            for z in os.listdir("./"):
                if(".pickle" in z or ".txt" in z):
                    move(z,newpath)

        sleep(5)
    sleep(5)
print("Hopefully Everything will Work, Sairam Swami Meere Dhikku")
os.system("poweroff")
            
