#!/bin/sh

export ROS_MASTER_URI=http://localhost:11333
export GAZEBO_MASTER_URI=http://localhost:11334

OUT_PATH="rl/out_dir/models"
EXPERIMENT=36
START=0
PER="false"
HER="false"

nohup roslaunch quadruped quadruped_control.launch \
    >> $OUT_PATH/exp$EXPERIMENT/ros.log &
nohup python3 -u rddpg_v2.py \
    --experiment $EXPERIMENT \
    --out_path $OUT_PATH \
    --start $START \
    --per $PER \
    --her $HER \
    >> $OUT_PATH/exp$EXPERIMENT/learn.log &
