#!/bin/sh

export ROS_MASTER_URI=http://localhost:11317
export GAZEBO_MASTER_URI=http://localhost:11318

OUT_PATH="weights/actor_pretrain"
EXPERIMENT=53
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
