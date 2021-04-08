#!/bin/sh

export ROS_MASTER_URI=http://localhost:11353
export GAZEBO_MASTER_URI=http://localhost:11343

OUT_PATH="rl/out_dir/models/ars"
EXPERIMENT=3
START=0

nohup roslaunch quadruped quadruped_control.launch \
    >> $OUT_PATH/exp$EXPERIMENT/ros.log &
nohup python3 -u ars.py \
    --experiment $EXPERIMENT \
    --out_path $OUT_PATH \
    --start $START \
    >> $OUT_PATH/exp$EXPERIMENT/learn.log &
