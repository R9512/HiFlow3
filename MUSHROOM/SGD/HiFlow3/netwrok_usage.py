
from scapy.all import sniff, TCP
import os,pickle
from datetime import datetime as dt
import time
# List of ports to monitor
PORTS = [8545, 5001, 45045, 44601, 44602, 44603, 44604, 44605,9512]

# Usage dictionary
USAGE = {port: {"IN": 0, "OUT": 0} for port in PORTS}
REGISTERED = False
def monitor_packet(packet):
    global USAGE,REGISTERED
    if packet.haslayer(TCP):
        sport = packet[TCP].sport
        dport = packet[TCP].dport
        length = len(packet)
        if(dport == 9512):
            save_stuff()


        # Outgoing traffic from the monitored port
        if sport in PORTS:
            USAGE[sport]["OUT"] += length

        # Incoming traffic to the monitored port
        if dport in PORTS:
            USAGE[dport]["IN"] += length

        
def save_stuff():
    global USAGE,REGISTERED
    if(USAGE[5001]["IN"] == 0):
        return

    print("\n=== Traffic Usage ===")
    with open('network.txt') as fp:
        for line in fp:
            if(len(line) == 0):
                continue
            port,data = line.split(",")
            port = int(port)
            data = int(data[:-1])
            USAGE[port]["OUT"] -= data
    
    usage = {"IPFS":0,"BLOCKCHAIN":0,"COORDINATOR":0,"VALIDATORS":{"IN": 0, "OUT": 0}}
    for port in PORTS:
        if(port == 5001):
            usage["IPFS"] = USAGE[port]
        elif(port == 8545):
            usage["BLOCKCHAIN"] = USAGE[port]
        elif(port == 45045):
            usage["COORDINATOR"] = USAGE[port]
        else:
            usage["VALIDATORS"]["IN"]+= USAGE[port]["IN"]
            usage["VALIDATORS"]["OUT"]+= USAGE[port]["OUT"]

        print(f"Port {port}: IN={USAGE[port]['IN']} bytes, OUT={USAGE[port]['OUT']} bytes")

    print("HERE")
    if(os.path.exists("./communication.pickle")):
        print("afdg")
        with (open("communication.pickle", "rb")) as openfile:
            print("asdg")
            data = pickle.load(openfile)
            print(data)
        print("Gathere Data",data)
        data[str(dt.now())] = usage.copy()
        with (open("communication.pickle", "wb")) as openfile:
            pickle.dump(data,openfile)
    else:
        print("here")
        to_dump = {}
        to_dump[str(dt.now())] = usage.copy()
        with (open("communication.pickle", "wb")) as openfile:
            pickle.dump(to_dump,openfile)

    USAGE = {port: {"IN": 0, "OUT": 0} for port in PORTS}
    #time.sleep(1)


# Catch Ctrl+C and kill signals


# BPF filter for all monitored ports
BPF_FILTER = " or ".join([f"tcp port {p}" for p in PORTS])
print("Started the Sniffifing")
sniff(filter=BPF_FILTER, prn=monitor_packet, store=0, iface="lo")
