#!/bin/sh

export ROS_MASTER_URI=http://localhost:11359
export GAZEBO_MASTER_URI=http://localhost:11360

roslaunch quadruped quadruped_control.launch
