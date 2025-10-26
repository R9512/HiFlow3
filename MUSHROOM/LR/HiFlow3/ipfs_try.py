import ipfs_api as ips
import pickle
import gnupg
gpg = gnupg.GPG()
with open("model.pickle","rb") as f:
    model = pickle.load(f)
serialized_data = pickle.dumps(model)
encrypted_data = gpg.encrypt(serialized_data,"sender@sairam.com")
print("Here is encypted data:\n",encrypted_data)
with open("e_model.pkl", "wb") as file:
    file.write(encrypted_data.data)
cid = ips.publish("e_model.pkl")
print(cid)