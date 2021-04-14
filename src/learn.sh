#!/bin/sh

export ROS_MASTER_URI=http://localhost:11344
export GAZEBO_MASTER_URI=http://localhost:11345

OUT_PATH="rl/out_dir/models"
<<<<<<< HEAD
EXPERIMENT=19
START=1375
=======
EXPERIMENT=21
START=0
>>>>>>> 7f519d57fc7155cebdbf32308d02f4109026bdd6
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
