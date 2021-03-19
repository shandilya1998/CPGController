#!/bin/sh

nohup roslaunch quadruped quadruped_control.launch >> ros.log &
nohup python3 -u learn.py >> learn.log &
