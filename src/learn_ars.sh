#!/bin/sh

export ROS_MASTER_URI=http://localhost:11351
export GAZEBO_MASTER_URI=http://localhost:11341

OUT_PATH="rl/out_dir/models/ars"
EXPERIMENT=2
START=75

nohup roslaunch quadruped quadruped_control.launch \
    >> $OUT_PATH/exp$EXPERIMENT/ros.log &
nohup python3 -u ars.py \
    --experiment $EXPERIMENT \
    --out_path $OUT_PATH \
    >> $OUT_PATH/exp$EXPERIMENT/learn.log &
