#!/bin/sh

export ROS_MASTER_URI=http://localhost:11344
export GAZEBO_MASTER_URI=http://localhost:11345

OUT_PATH="rl/out_dir/models"
EXPERIMENT=21
START=0
PER="false"
HER="true"

nohup roslaunch quadruped quadruped_control.launch \
    >> $OUT_PATH/exp$EXPERIMENT/ros.log &
nohup python3 -u learn.py \
    --experiment $EXPERIMENT \
    --out_path $OUT_PATH \
    --start $START \
    --per $PER \
    --her $HER \
    >> $OUT_PATH/exp$EXPERIMENT/learn.log &
