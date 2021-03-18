#!/bin/sh
stdbuf -oL roslaunch quadruped quadruped_control.launch > learn.log &
stdbuf -oL python3 learn.py > learn.log &
