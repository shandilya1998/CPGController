#!/bin/sh

export ROS_MASTER_URI=http://localhost:11313
export GAZEBO_MASTER_URI=http://localhost:11314

OUT_PATH="rl/out_dir/models"
EXPERIMENT=14
START=1950

nohup roslaunch quadruped quadruped_control.launch \
    >> $OUT_PATH/exp$EXPERIMENT/ros.log &
nohup python3 -u learn.py \
    --experiment $EXPERIMENT \
    --out_path $OUT_PATH \
    --start $START \
    >> $OUT_PATH/exp$EXPERIMENT/learn.log &
