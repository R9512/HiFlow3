#!/bin/bash
rm done_part.flag
echo "Starting server"
python3 server.py &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 1 15`; do
    echo "Starting client $i"
    python3 client.py $i &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait_time=0
while [ ! -f "done_part.flag" ]
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
