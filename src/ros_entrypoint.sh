#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
echo "export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/app/simulations/ws/src" >> ~/.bashrc
source ~/.bashrc
echo "source /opt/ros/noetic/setup.bash" >> ~/.bash_profile
echo "export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/app/simulations/ws/src" >> ~/.bash_profile
cd simulations/ws && catkin_make
exec "$@"
