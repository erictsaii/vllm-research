#!/bin/bash 

sudo tc qdisc del dev enp1s0f0 root 2>/dev/null
sudo tc qdisc replace dev enp1s0f0 root tbf rate 777mbit burst 1m latency 50ms