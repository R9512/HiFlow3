import gnupg
gpg = gnupg.GPG(gnupghome='/home/swami/.gnupg/')
gpg.import_keys_file("../keys/DECENTRALIZE_AGGREGATOR/private.pem")
recipients = []
for x in range(1,4):
    recipients.append("VALIDATOR_"+str(x)+"@HiFlow3.com")
print(recipients)
encrypted_data = gpg.encrypt("temp",recipients,sign="DECENTRALIZE_AGGREGATOR@HiFlow3.com")
print(encrypted_data.data)