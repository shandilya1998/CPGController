#!/bin/sh

OUT_PATH="weights/actor_pretrain"
EXPERIMENT=18

nohup roslaunch quadruped quadruped_control.launch \
    >> $OUT_PATH/exp$EXPERIMENT/ros.log &
nohup python3 -u learn.py \
    --experiment $EXPERIMENT \
    --out_path $OUT_PATH \
    >> $OUT_PATH/exp$EXPERIMENT/learn.log &
