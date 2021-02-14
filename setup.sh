#!/bin/bash
pip3 install virtualenv
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
apt-get update
sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -
apt update
apt install ros-melodic-desktop-full
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
apt install python-rosdep
rosdep init
rosdep update
cd src/simulations/ws
sed -i 's#/home/shandilya/Desktop/CNS/DDP#/content/CPGController#g' build/CMakeCache.txt
sed -i 's#/home/shandilya/Desktop/CNS/DDP#/content/CPGController#g' build/Makefile
catkin_make
source devel/setup.bash
cd .. && cd ..
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/content/CPGController/src/simulations/ws/src
