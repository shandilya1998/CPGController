#!/bin/sh

export ROS_MASTER_URI=http://localhost:11352
export GAZEBO_MASTER_URI=http://localhost:11353

OUT_PATH="rl/out_dir/models"
EXPERIMENT=12
START=25

nohup roslaunch quadruped quadruped_control.launch \
    >> $OUT_PATH/exp$EXPERIMENT/ros.log &
nohup python3 -u learn.py \
    --experiment $EXPERIMENT \
    --out_path $OUT_PATH \
    --start $START \
    >> $OUT_PATH/exp$EXPERIMENT/learn.log &
