 
import pickle
import os

with open("old_communication.pickle","rb") as f:
    a = pickle.load(f)
b = list(a.keys())
b = b[2:]
c = {}
for x in b:
    c[x] = a[x]
with open("communication.pickle","wb") as f:
    pickle.dump(c,f)
print(len(c))
