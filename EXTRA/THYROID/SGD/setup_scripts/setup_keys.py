"""
    - Has all the crypto functions wrapped in to one.
"""
from os import mkdir
from shutil import rmtree
import tomli
import gnupg
gpg = gnupg.GPG()#Need to give home path
gpg.encoding = 'utf-8'
print(gpg)
from os import mkdir
from shutil import rmtree
import tomli
import gnupg
gpg = gnupg.GPG(gnupghome='/home/swami/.gnupg/')#Need to give home path
gpg.encoding = 'utf-8'

def create_new_key(name):
    
    """
        Generate New Keys, save the private key and upload the publick key to IPFS
    """
    BASE_PATH = "../keys/"+name+"/" 
    mkdir(BASE_PATH)
    input_data = gpg.gen_key_input(key_type="RSA", key_length=1024,name_email=name+"@HiFlow3.com",no_protection=True)
    key = gpg.gen_key(input_data)
    ascii_armored_public_keys = gpg.export_keys(name,output=BASE_PATH+"public.pem",expect_passphrase=False)
    ascii_armored_private_keys = gpg.export_keys(name,True,output=BASE_PATH+"private.pem", expect_passphrase=False)

def create_new_keystore():
    """
        - Creates a New Set of Key and Store it in the keys folder
        - Each manager will have his own set of keys stored with his ID
        - So does every client.
    """
    rmtree("../keys",ignore_errors=True)
    mkdir("../keys")#Deleting the existing one and creating the new one
    NUM_MANAGER = 5
    NUM_NODES = 50
   
    create_new_key("DECENTRALIZE_AGGREGATOR")
    for x in range(0,NUM_NODES):
        create_new_key("NODE_"+str(x+1))
    for x in range(0,NUM_MANAGER):
        create_new_key("VALIDATOR_"+str(x+1))

if __name__ =="__main__":
    create_new_keystore()
