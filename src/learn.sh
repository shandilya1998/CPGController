#!/bin/sh

export ROS_MASTER_URI=http://localhost:11325
export GAZEBO_MASTER_URI=http://localhost:11326

OUT_PATH="rl/out_dir/models"
EXPERIMENT=13
START=475

nohup roslaunch quadruped quadruped_control.launch \
    >> $OUT_PATH/exp$EXPERIMENT/ros.log &
nohup python3 -u learn.py \
    --experiment $EXPERIMENT \
    --out_path $OUT_PATH \
    --start $START \
    >> $OUT_PATH/exp$EXPERIMENT/learn.log &
