import toml
import os
from datetime import datetime
from time import sleep
from shutil import move

ITER=10
CONFIG_PATH = "./config.toml"
zeta = 0
for zeta in range(0,1):
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
            count = 0
            while count < ITER:
                os.system("./run.sh")

                sleep(3)
                count+=1
            #Add codesuch a way that it does deletesmodels and csv
            newpath = "./models_zeta="+str(zeta)+"/"+str(current)
            os.mkdir(newpath)
            for z in os.listdir("./"):
                if(".pickle" in z):
                    move(z,newpath)

        sleep(5)
    sleep(5)
