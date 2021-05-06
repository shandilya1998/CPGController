export ROS_MASTER_URI=http://localhost:11315
export GAZEBO_MASTER_URI=http://localhost:11316

OUT_PATH="rl/out_dir/models"
EXPERIMENT=51

nohup roslaunch quadruped quadruped_control.launch \
    >> $OUT_PATH/exp$EXPERIMENT/ros.log &
nohup python3 -u rddpg_torch.py \
    --experiment $EXPERIMENT \
    --out_path $OUT_PATH >> $OUT_PATH/exp$EXPERIMENT/learn.log &
