#!/bin/sh

export ROS_MASTER_URI=http://localhost:11360
export GAZEBO_MASTER_URI=http://localhost:11361

OUT_PATH="rl/out_dir/models"
EXPERIMENT=12
START=0

nohup roslaunch quadruped quadruped_control.launch \
    >> $OUT_PATH/exp$EXPERIMENT/ros.log &
nohup python3 -u learn.py \
    --experiment $EXPERIMENT \
    --out_path $OUT_PATH \
    --start $START \
    >> $OUT_PATH/exp$EXPERIMENT/learn.log &
