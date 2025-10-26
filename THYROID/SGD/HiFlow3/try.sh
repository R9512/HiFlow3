#!/bin/bash


rm signal.flag
rm done.flag
rm -rf ./node_data/
rm network.txt    #Just remove netwrok
clear
num_validators=${1:-5}
num_clients=${2:-3}
portnum=${3:-44601}
uv run partitioner.py
uv run  coordinator.py $portnum  &

mkdir node_data
sleep 3
num_node=1
for i in `seq 1 $num_validators`; do
    echo "[SHELL]Starting Validator $i  @ $portnum"   
    #konsole --noclose -e "uv run validator.py $portnum  $i &" &
    uv run validator.py $portnum  $i & #Should Add ID Here
    sleep 3
    for j in `seq 1 $num_clients`; do
        echo "[SHELL]      Starting Node $num_node "
         uv run node.py $portnum  $num_node & #Should Add ID Here
#         #konsole --noclose -e "python3 node.py $portnum" &
        num_node=$((num_node +1))
        sleep 1
    done
    portnum=$((portnum+1)) 
done
wait_time=0
while [ ! -f "signal.flag" ];
do
    sleep 1
    wait_time=$((wait_time +1))
    if [ $wait_time -ge 200 ];
    then
        kill $(jobs -p)
        sleep 7
        exit 1
    fi
done
kill $(jobs -p)

#Send a TCP ping here so that the guy can realize and stop logging
rm done.flag
telnet 127.0.0.1 9512 #For closing the log

sleep 3
rm network.txt




