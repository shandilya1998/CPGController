#!/bin/sh

export ROS_MASTER_URI=http://localhost:11319
export GAZEBO_MASTER_URI=http://localhost:11320

OUT_PATH="weights/actor_pretrain"
EXPERIMENT=33
START=0
PER="false"
HER="false"

nohup roslaunch quadruped quadruped_control.launch \
    >> $OUT_PATH/exp$EXPERIMENT/ros.log &
nohup python3 -u rddpg.py \
    --experiment $EXPERIMENT \
    --out_path $OUT_PATH \
    --start $START \
    --per $PER \
    --her $HER \
    >> $OUT_PATH/exp$EXPERIMENT/learn.log &
