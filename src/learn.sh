#!/bin/sh

export ROS_MASTER_URI=http://localhost:11311
export GAZEBO_MASTER_URI=http://localhost:11312

OUT_PATH="weights/actor_pretrain"
EXPERIMENT=27
START=0

nohup roslaunch quadruped quadruped_control.launch \
    >> $OUT_PATH/exp$EXPERIMENT/ros.log &
nohup python3 -u learn.py \
    --experiment $EXPERIMENT \
    --out_path $OUT_PATH \
    --start $START \
    >> $OUT_PATH/exp$EXPERIMENT/learn.log &
